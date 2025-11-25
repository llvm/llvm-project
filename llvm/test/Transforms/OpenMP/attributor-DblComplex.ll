; RUN: opt --mtriple=amdgcn-amd-amdhsa -S -passes='attributor' < %s | FileCheck %s

; verify that the following test case does not assert in the attributor due
; to addrspace 5 to generic casts seen when compiling for amdgcn-amd-amdhsa
;
; clang++ -O2 -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -Xopenmp-target=amdgcn-amd-amdhsa -march=gfx908 red-DblComplex.cpp
;
; #include <complex>
; std::complex<double> reduce(std::complex<double> dres[], int n) {
;     std::complex<double> dinp(0.0, 0.0);
;     #pragma omp target teams distribute parallel for map(to: dres) map(tofrom:dinp) reduction(+:dinp)
;     for (int i = 0; i < n; i++) {
;         dinp += dres[i];
;     }
;     return(dinp);
; }

; CHECK: define internal void @_omp_reduction_shuffle_and_reduce_func

; ModuleID = 'clang-red-DblComplex-openmp-amdgcn-amd-amdhsa-gfx908.bc'
source_filename = "clang-red-DblComplex.cpp"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

%struct.ident_t = type { i32, i32, i32, i32, ptr }
%struct.DynamicEnvironmentTy = type { i16 }
%struct.KernelEnvironmentTy = type { %struct.ConfigurationEnvironmentTy, ptr, ptr }
%struct.ConfigurationEnvironmentTy = type { i8, i8, i8, i32, i32, i32, i32, i32, i32 }
%"struct.std::complex" = type { { double, double } }
%struct._globalized_locals_ty = type { %"struct.std::complex" }

@__omp_plugin_enable_fast_reduction = weak addrspace(1) constant i8 0
@__omp_rtl_debug_kind = weak_odr hidden local_unnamed_addr addrspace(1) constant i32 0
@__omp_rtl_assume_teams_oversubscription = weak_odr hidden local_unnamed_addr addrspace(1) constant i32 0
@__omp_rtl_assume_threads_oversubscription = weak_odr hidden local_unnamed_addr addrspace(1) constant i32 0
@__omp_rtl_assume_no_thread_state = weak_odr hidden local_unnamed_addr addrspace(1) constant i32 0
@__omp_rtl_assume_no_nested_parallelism = weak_odr hidden local_unnamed_addr addrspace(1) constant i32 0
@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr addrspace(1) constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @0 }, align 8
@__omp_offloading_fd00_426262e_main_l15_dynamic_environment = weak_odr protected addrspace(1) global %struct.DynamicEnvironmentTy zeroinitializer
@__omp_offloading_fd00_426262e_main_l15_kernel_environment = weak_odr protected addrspace(1) constant %struct.KernelEnvironmentTy { %struct.ConfigurationEnvironmentTy { i8 0, i8 1, i8 2, i32 1, i32 256, i32 0, i32 0, i32 16, i32 1024 }, ptr addrspacecast (ptr addrspace(1) @1 to ptr), ptr addrspacecast (ptr addrspace(1) @__omp_offloading_fd00_426262e_main_l15_dynamic_environment to ptr) }
@2 = private unnamed_addr addrspace(1) constant %struct.ident_t { i32 0, i32 2050, i32 0, i32 22, ptr @0 }, align 8
@3 = private unnamed_addr addrspace(1) constant %struct.ident_t { i32 0, i32 514, i32 0, i32 22, ptr @0 }, align 8
@__openmp_nvptx_data_transfer_temporary_storage = weak addrspace(3) global [64 x i32] undef
@4 = private unnamed_addr addrspace(1) constant %struct.ident_t { i32 0, i32 66, i32 0, i32 22, ptr @0 }, align 8
@__omp_offloading_fd00_426262e_main_l15_wg_size = weak addrspace(1) constant i16 256
@__omp_offloading_fd00_426262e_main_l15_exec_mode = weak addrspace(1) constant i8 2
@__oclc_ABI_version = weak_odr hidden local_unnamed_addr addrspace(4) constant i32 500
@llvm.compiler.used = appending addrspace(1) global [4 x ptr] [ptr addrspacecast (ptr addrspace(1) @__omp_offloading_fd00_426262e_main_l15_exec_mode to ptr), ptr addrspacecast (ptr addrspace(1) @__omp_offloading_fd00_426262e_main_l15_wg_size to ptr), ptr addrspacecast (ptr addrspace(1) @__omp_plugin_enable_fast_reduction to ptr), ptr addrspacecast (ptr addrspace(3) @__openmp_nvptx_data_transfer_temporary_storage to ptr)], section "llvm.metadata"

; Function Attrs: alwaysinline norecurse nounwind
define weak_odr protected amdgpu_kernel void @__omp_offloading_fd00_426262e_main_l15(ptr noalias noundef %dyn_ptr, ptr noundef nonnull align 8 dereferenceable(16) %dinp, ptr noundef nonnull align 8 dereferenceable(1600) %dres) local_unnamed_addr #0 {
entry:
  %dinp1.i = alloca %"struct.std::complex", align 8, addrspace(5)
  %.omp.comb.lb.i = alloca i32, align 4, addrspace(5)
  %.omp.comb.ub.i = alloca i32, align 4, addrspace(5)
  %.omp.stride.i = alloca i32, align 4, addrspace(5)
  %.omp.is_last.i = alloca i32, align 4, addrspace(5)
  %captured_vars_addrs.i = alloca [4 x ptr], align 8, addrspace(5)
  %.omp.reduction.red_list.i = alloca [1 x ptr], align 8, addrspace(5)
  %dinp.global1 = addrspacecast ptr %dinp to ptr addrspace(1)
  %0 = tail call i32 @__kmpc_target_init(ptr addrspacecast (ptr addrspace(1) @__omp_offloading_fd00_426262e_main_l15_kernel_environment to ptr), ptr %dyn_ptr) #2
  %exec_user_code = icmp eq i32 %0, -1
  br i1 %exec_user_code, label %user_code.entry, label %common.ret

common.ret:                                       ; preds = %entry, %__omp_offloading_fd00_426262e_main_l15_omp_outlined.exit
  ret void

user_code.entry:                                  ; preds = %entry
  %1 = tail call i32 @__kmpc_global_thread_num(ptr addrspacecast (ptr addrspace(1) @1 to ptr)) #2
  call void @llvm.lifetime.start.p5(i64 32, ptr addrspace(5) %captured_vars_addrs.i)
  call void @llvm.lifetime.start.p5(i64 8, ptr addrspace(5) %.omp.reduction.red_list.i)
  %dinp1.ascast.i = addrspacecast ptr addrspace(5) %dinp1.i to ptr
  %.omp.comb.lb.ascast.i = addrspacecast ptr addrspace(5) %.omp.comb.lb.i to ptr
  %.omp.comb.ub.ascast.i = addrspacecast ptr addrspace(5) %.omp.comb.ub.i to ptr
  %.omp.stride.ascast.i = addrspacecast ptr addrspace(5) %.omp.stride.i to ptr
  %.omp.is_last.ascast.i = addrspacecast ptr addrspace(5) %.omp.is_last.i to ptr
  %captured_vars_addrs.ascast.i = addrspacecast ptr addrspace(5) %captured_vars_addrs.i to ptr
  %.omp.reduction.red_list.ascast.i = addrspacecast ptr addrspace(5) %.omp.reduction.red_list.i to ptr
  call void @llvm.lifetime.start.p5(i64 16, ptr addrspace(5) %dinp1.i) #13, !noalias !9
  %_M_value.imagp.i.i = getelementptr inbounds i8, ptr addrspace(5) %dinp1.i, i32 8
  store double 0.000000e+00, ptr addrspace(5) %dinp1.i, align 8, !noalias !9
  store double 0.000000e+00, ptr addrspace(5) %_M_value.imagp.i.i, align 8, !noalias !9
  call void @llvm.lifetime.start.p5(i64 4, ptr addrspace(5) %.omp.comb.lb.i) #13, !noalias !9
  store i32 0, ptr addrspace(5) %.omp.comb.lb.i, align 4, !tbaa !12, !noalias !9
  call void @llvm.lifetime.start.p5(i64 4, ptr addrspace(5) %.omp.comb.ub.i) #13, !noalias !9
  store i32 99, ptr addrspace(5) %.omp.comb.ub.i, align 4, !tbaa !12, !noalias !9
  call void @llvm.lifetime.start.p5(i64 4, ptr addrspace(5) %.omp.stride.i) #13, !noalias !9
  store i32 1, ptr addrspace(5) %.omp.stride.i, align 4, !tbaa !12, !noalias !9
  call void @llvm.lifetime.start.p5(i64 4, ptr addrspace(5) %.omp.is_last.i) #13, !noalias !9
  store i32 0, ptr addrspace(5) %.omp.is_last.i, align 4, !tbaa !12, !noalias !9
  %nvptx_num_threads.i = tail call i32 @__kmpc_get_hardware_num_threads_in_block() #2, !noalias !9
  call void @__kmpc_distribute_static_init_4(ptr addrspacecast (ptr addrspace(1) @2 to ptr), i32 %1, i32 91, ptr nonnull %.omp.is_last.ascast.i, ptr nonnull %.omp.comb.lb.ascast.i, ptr nonnull %.omp.comb.ub.ascast.i, ptr nonnull %.omp.stride.ascast.i, i32 1, i32 %nvptx_num_threads.i) #2, !noalias !9
  %2 = load i32, ptr addrspace(5) %.omp.comb.ub.i, align 4, !noalias !9
  %cond.i = call i32 @llvm.smin.i32(i32 %2, i32 99)
  store i32 %cond.i, ptr addrspace(5) %.omp.comb.ub.i, align 4, !tbaa !12, !noalias !9
  %.omp.iv.012.i = load i32, ptr addrspace(5) %.omp.comb.lb.i, align 4, !noalias !9
  %cmp213.i = icmp slt i32 %.omp.iv.012.i, 100
  br i1 %cmp213.i, label %omp.inner.for.body.lr.ph.i, label %omp.loop.exit.i

omp.inner.for.body.lr.ph.i:                       ; preds = %user_code.entry
  %3 = getelementptr inbounds i8, ptr addrspace(5) %captured_vars_addrs.i, i32 8
  %4 = getelementptr inbounds i8, ptr addrspace(5) %captured_vars_addrs.i, i32 16
  %5 = getelementptr inbounds i8, ptr addrspace(5) %captured_vars_addrs.i, i32 24
  br label %omp.inner.for.body.i

omp.inner.for.body.i:                             ; preds = %omp.inner.for.body.i, %omp.inner.for.body.lr.ph.i
  %.omp.iv.015.i = phi i32 [ %.omp.iv.012.i, %omp.inner.for.body.lr.ph.i ], [ %add3.i, %omp.inner.for.body.i ]
  %storemerge14.i = phi i32 [ %cond.i, %omp.inner.for.body.lr.ph.i ], [ %cond9.i, %omp.inner.for.body.i ]
  %6 = zext i32 %.omp.iv.015.i to i64
  %7 = zext i32 %storemerge14.i to i64
  %8 = inttoptr i64 %6 to ptr
  store ptr %8, ptr addrspace(5) %captured_vars_addrs.i, align 8, !tbaa !16, !noalias !9
  %9 = inttoptr i64 %7 to ptr
  store ptr %9, ptr addrspace(5) %3, align 8, !tbaa !16, !noalias !9
  store ptr %dinp1.ascast.i, ptr addrspace(5) %4, align 8, !tbaa !16, !noalias !9
  store ptr %dres, ptr addrspace(5) %5, align 8, !tbaa !16, !noalias !9
  call void @__kmpc_parallel_51(ptr addrspacecast (ptr addrspace(1) @1 to ptr), i32 %1, i32 1, i32 -1, i32 -1, ptr nonnull @__omp_offloading_fd00_426262e_main_l15_omp_outlined_omp_outlined, ptr null, ptr nonnull %captured_vars_addrs.ascast.i, i64 4) #2, !noalias !9
  %10 = load i32, ptr addrspace(5) %.omp.stride.i, align 4, !tbaa !12, !noalias !9
  %11 = load i32, ptr addrspace(5) %.omp.comb.lb.i, align 4, !tbaa !12, !noalias !9
  %add3.i = add nsw i32 %11, %10
  store i32 %add3.i, ptr addrspace(5) %.omp.comb.lb.i, align 4, !tbaa !12, !noalias !9
  %12 = load i32, ptr addrspace(5) %.omp.comb.ub.i, align 4, !tbaa !12, !noalias !9
  %add4.i = add nsw i32 %12, %10
  %cond9.i = call i32 @llvm.smin.i32(i32 %add4.i, i32 99)
  store i32 %cond9.i, ptr addrspace(5) %.omp.comb.ub.i, align 4, !tbaa !12, !noalias !9
  %cmp2.i = icmp slt i32 %add3.i, 100
  br i1 %cmp2.i, label %omp.inner.for.body.i, label %omp.loop.exit.i

omp.loop.exit.i:                                  ; preds = %omp.inner.for.body.i, %user_code.entry
  call void @__kmpc_distribute_static_fini(ptr addrspacecast (ptr addrspace(1) @2 to ptr), i32 %1) #2, !noalias !9
  call void @llvm.lifetime.end.p5(i64 4, ptr addrspace(5) %.omp.is_last.i) #2, !noalias !9
  call void @llvm.lifetime.end.p5(i64 4, ptr addrspace(5) %.omp.stride.i) #2, !noalias !9
  call void @llvm.lifetime.end.p5(i64 4, ptr addrspace(5) %.omp.comb.ub.i) #2, !noalias !9
  call void @llvm.lifetime.end.p5(i64 4, ptr addrspace(5) %.omp.comb.lb.i) #2, !noalias !9
  store ptr %dinp1.ascast.i, ptr addrspace(5) %.omp.reduction.red_list.i, align 8, !noalias !9
  %"_openmp_teams_reductions_buffer_$_$ptr.i" = call ptr @__kmpc_reduction_get_fixed_buffer() #2, !noalias !9
  %13 = call i32 @__kmpc_nvptx_teams_reduce_nowait_v2(ptr addrspacecast (ptr addrspace(1) @1 to ptr), ptr %"_openmp_teams_reductions_buffer_$_$ptr.i", i32 1024, i64 16, ptr nonnull %.omp.reduction.red_list.ascast.i, ptr nonnull @_omp_reduction_shuffle_and_reduce_func.1, ptr nonnull @_omp_reduction_inter_warp_copy_func.2, ptr nonnull @_omp_reduction_list_to_global_copy_func, ptr nonnull @_omp_reduction_list_to_global_reduce_func, ptr nonnull @_omp_reduction_global_to_list_copy_func, ptr nonnull @_omp_reduction_global_to_list_reduce_func) #2, !noalias !9
  %14 = icmp eq i32 %13, 1
  br i1 %14, label %.omp.reduction.then.i, label %__omp_offloading_fd00_426262e_main_l15_omp_outlined.exit

.omp.reduction.then.i:                            ; preds = %omp.loop.exit.i
  %_M_value.real.i.i.i = load double, ptr addrspace(5) %dinp1.i, align 8, !noalias !9
  %_M_value.imag.i.i.i = load double, ptr addrspace(5) %_M_value.imagp.i.i, align 8, !noalias !9
  %_M_value.real.i.i = load double, ptr addrspace(1) %dinp.global1, align 8, !noalias !9
  %_M_value.imagp.i11.i = getelementptr inbounds i8, ptr addrspace(1) %dinp.global1, i64 8
  %_M_value.imag.i.i = load double, ptr addrspace(1) %_M_value.imagp.i11.i, align 8, !noalias !9
  %add.r.i.i = fadd double %_M_value.real.i.i.i, %_M_value.real.i.i
  %add.i.i.i = fadd double %_M_value.imag.i.i.i, %_M_value.imag.i.i
  store double %add.r.i.i, ptr addrspace(1) %dinp.global1, align 8, !noalias !9
  store double %add.i.i.i, ptr addrspace(1) %_M_value.imagp.i11.i, align 8, !noalias !9
  br label %__omp_offloading_fd00_426262e_main_l15_omp_outlined.exit

__omp_offloading_fd00_426262e_main_l15_omp_outlined.exit: ; preds = %omp.loop.exit.i, %.omp.reduction.then.i
  call void @llvm.lifetime.end.p5(i64 16, ptr addrspace(5) %dinp1.i) #2, !noalias !9
  call void @llvm.lifetime.end.p5(i64 32, ptr addrspace(5) %captured_vars_addrs.i)
  call void @llvm.lifetime.end.p5(i64 8, ptr addrspace(5) %.omp.reduction.red_list.i)
  call void @__kmpc_target_deinit() #2
  br label %common.ret
}

declare i32 @__kmpc_target_init(ptr, ptr) local_unnamed_addr

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p5(i64 immarg, ptr addrspace(5) nocapture) #1

; Function Attrs: nounwind
declare i32 @__kmpc_get_hardware_num_threads_in_block() local_unnamed_addr #2

; Function Attrs: nounwind
declare void @__kmpc_distribute_static_init_4(ptr, i32, i32, ptr, ptr, ptr, ptr, i32, i32) local_unnamed_addr #2

; Function Attrs: alwaysinline norecurse nounwind
define internal void @__omp_offloading_fd00_426262e_main_l15_omp_outlined_omp_outlined(ptr noalias nocapture noundef readonly %.global_tid., ptr noalias nocapture readnone %.bound_tid., i64 noundef %.previous.lb., i64 noundef %.previous.ub., ptr nocapture noundef nonnull align 8 dereferenceable(16) %dinp, ptr nocapture noundef nonnull readonly align 8 dereferenceable(1600) %dres) #3 {
entry:
  %.omp.lb = alloca i32, align 4, addrspace(5)
  %.omp.ub = alloca i32, align 4, addrspace(5)
  %.omp.stride = alloca i32, align 4, addrspace(5)
  %.omp.is_last = alloca i32, align 4, addrspace(5)
  %dinp2 = alloca %"struct.std::complex", align 8, addrspace(5)
  %.omp.reduction.red_list = alloca [1 x ptr], align 8, addrspace(5)
  %.omp.lb.ascast = addrspacecast ptr addrspace(5) %.omp.lb to ptr
  %.omp.ub.ascast = addrspacecast ptr addrspace(5) %.omp.ub to ptr
  %.omp.stride.ascast = addrspacecast ptr addrspace(5) %.omp.stride to ptr
  %.omp.is_last.ascast = addrspacecast ptr addrspace(5) %.omp.is_last to ptr
  %dinp2.ascast = addrspacecast ptr addrspace(5) %dinp2 to ptr
  %.omp.reduction.red_list.ascast = addrspacecast ptr addrspace(5) %.omp.reduction.red_list to ptr
  call void @llvm.lifetime.start.p5(i64 4, ptr addrspace(5) %.omp.lb) #2
  call void @llvm.lifetime.start.p5(i64 4, ptr addrspace(5) %.omp.ub) #2
  %conv = trunc i64 %.previous.lb. to i32
  %conv1 = trunc i64 %.previous.ub. to i32
  store i32 %conv, ptr addrspace(5) %.omp.lb, align 4, !tbaa !12
  store i32 %conv1, ptr addrspace(5) %.omp.ub, align 4, !tbaa !12
  call void @llvm.lifetime.start.p5(i64 4, ptr addrspace(5) %.omp.stride) #2
  store i32 1, ptr addrspace(5) %.omp.stride, align 4, !tbaa !12
  call void @llvm.lifetime.start.p5(i64 4, ptr addrspace(5) %.omp.is_last) #2
  store i32 0, ptr addrspace(5) %.omp.is_last, align 4, !tbaa !12
  call void @llvm.lifetime.start.p5(i64 16, ptr addrspace(5) %dinp2) #2
  %_M_value.imagp.i = getelementptr inbounds i8, ptr addrspace(5) %dinp2, i32 8
  %0 = load i32, ptr %.global_tid., align 4, !tbaa !12
  call void @__kmpc_for_static_init_4(ptr addrspacecast (ptr addrspace(1) @3 to ptr), i32 %0, i32 33, ptr nonnull %.omp.is_last.ascast, ptr nonnull %.omp.lb.ascast, ptr nonnull %.omp.ub.ascast, ptr nonnull %.omp.stride.ascast, i32 1, i32 1) #2
  %1 = load i32, ptr addrspace(5) %.omp.lb, align 4, !tbaa !12
  %conv320 = sext i32 %1 to i64
  %cmp.not21 = icmp ugt i64 %conv320, %.previous.ub.
  br i1 %cmp.not21, label %omp.loop.exit, label %omp.inner.for.body.lr.ph

omp.inner.for.body.lr.ph:                         ; preds = %entry
  %2 = load i32, ptr addrspace(5) %.omp.stride, align 4, !tbaa !12
  br label %omp.inner.for.body

omp.inner.for.body:                               ; preds = %omp.inner.for.body.lr.ph, %omp.inner.for.body
  %conv325 = phi i64 [ %conv320, %omp.inner.for.body.lr.ph ], [ %conv3, %omp.inner.for.body ]
  %_M_value.real.i1823 = phi double [ 0.000000e+00, %omp.inner.for.body.lr.ph ], [ %add.r.i, %omp.inner.for.body ]
  %add.i.i1922 = phi double [ 0.000000e+00, %omp.inner.for.body.lr.ph ], [ %add.i.i, %omp.inner.for.body ]
  %indvars = trunc i64 %conv325 to i32
  %arrayidx = getelementptr inbounds [100 x %"struct.std::complex"], ptr %dres, i64 0, i64 %conv325
  %_M_value.real.i.i = load double, ptr %arrayidx, align 8
  %_M_value.imagp.i.i = getelementptr inbounds i8, ptr %arrayidx, i64 8
  %_M_value.imag.i.i = load double, ptr %_M_value.imagp.i.i, align 8
  %add.r.i = fadd double %_M_value.real.i1823, %_M_value.real.i.i
  %add.i.i = fadd double %add.i.i1922, %_M_value.imag.i.i
  %add4 = add nsw i32 %2, %indvars
  %conv3 = sext i32 %add4 to i64
  %cmp.not = icmp ugt i64 %conv3, %.previous.ub.
  br i1 %cmp.not, label %omp.loop.exit, label %omp.inner.for.body

omp.loop.exit:                                    ; preds = %omp.inner.for.body, %entry
  %add.i.i19.lcssa = phi double [ 0.000000e+00, %entry ], [ %add.i.i, %omp.inner.for.body ]
  %_M_value.real.i18.lcssa = phi double [ 0.000000e+00, %entry ], [ %add.r.i, %omp.inner.for.body ]
  store double %_M_value.real.i18.lcssa, ptr addrspace(5) %dinp2, align 8
  store double %add.i.i19.lcssa, ptr addrspace(5) %_M_value.imagp.i, align 8
  call void @__kmpc_for_static_fini(ptr addrspacecast (ptr addrspace(1) @3 to ptr), i32 %0) #2
  store ptr %dinp2.ascast, ptr addrspace(5) %.omp.reduction.red_list, align 8
  %3 = call i32 @__kmpc_nvptx_parallel_reduce_nowait_v2(ptr addrspacecast (ptr addrspace(1) @1 to ptr), i64 16, ptr nonnull %.omp.reduction.red_list.ascast, ptr nonnull @_omp_reduction_shuffle_and_reduce_func, ptr nonnull @_omp_reduction_inter_warp_copy_func) #2
  %4 = icmp eq i32 %3, 1
  br i1 %4, label %.omp.reduction.then, label %.omp.reduction.done

.omp.reduction.then:                              ; preds = %omp.loop.exit
  %_M_value.real.i.i10 = load double, ptr addrspace(5) %dinp2, align 8
  %_M_value.imag.i.i12 = load double, ptr addrspace(5) %_M_value.imagp.i, align 8
  %_M_value.real.i13 = load double, ptr %dinp, align 8
  %_M_value.imagp.i14 = getelementptr inbounds i8, ptr %dinp, i64 8
  %_M_value.imag.i15 = load double, ptr %_M_value.imagp.i14, align 8
  %add.r.i16 = fadd double %_M_value.real.i.i10, %_M_value.real.i13
  %add.i.i17 = fadd double %_M_value.imag.i.i12, %_M_value.imag.i15
  store double %add.r.i16, ptr %dinp, align 8
  store double %add.i.i17, ptr %_M_value.imagp.i14, align 8
  br label %.omp.reduction.done

.omp.reduction.done:                              ; preds = %.omp.reduction.then, %omp.loop.exit
  call void @llvm.lifetime.end.p5(i64 16, ptr addrspace(5) %dinp2) #2
  call void @llvm.lifetime.end.p5(i64 4, ptr addrspace(5) %.omp.is_last) #2
  call void @llvm.lifetime.end.p5(i64 4, ptr addrspace(5) %.omp.stride) #2
  call void @llvm.lifetime.end.p5(i64 4, ptr addrspace(5) %.omp.ub) #2
  call void @llvm.lifetime.end.p5(i64 4, ptr addrspace(5) %.omp.lb) #2
  ret void
}

; Function Attrs: nounwind
declare void @__kmpc_for_static_init_4(ptr, i32, i32, ptr, ptr, ptr, ptr, i32, i32) local_unnamed_addr #2

; Function Attrs: nounwind
declare void @__kmpc_for_static_fini(ptr, i32) local_unnamed_addr #2

; Function Attrs: norecurse nounwind
define internal void @_omp_reduction_shuffle_and_reduce_func(ptr nocapture noundef readonly %0, i16 noundef signext %1, i16 noundef signext %2, i16 noundef signext %3) #4 {
entry:
  %4 = load ptr, ptr %0, align 8
  %5 = load i64, ptr %4, align 8
  %6 = tail call i32 @__kmpc_get_warp_size() #2
  %7 = trunc i32 %6 to i16
  %8 = tail call i64 @__kmpc_shuffle_int64(i64 %5, i16 %2, i16 %7) #2
  %9 = getelementptr i8, ptr %4, i64 8
  %10 = load i64, ptr %9, align 8
  %11 = tail call i32 @__kmpc_get_warp_size() #2
  %12 = trunc i32 %11 to i16
  %13 = tail call i64 @__kmpc_shuffle_int64(i64 %10, i16 %2, i16 %12) #2
  %14 = icmp eq i16 %3, 0
  %15 = icmp eq i16 %3, 1
  %16 = icmp ult i16 %1, %2
  %17 = and i1 %16, %15
  %18 = icmp eq i16 %3, 2
  %19 = and i16 %1, 1
  %20 = icmp eq i16 %19, 0
  %21 = and i1 %20, %18
  %22 = icmp sgt i16 %2, 0
  %23 = and i1 %22, %21
  %24 = or i1 %14, %17
  %25 = or i1 %24, %23
  br i1 %25, label %then, label %ifcont

then:                                             ; preds = %entry
  %26 = bitcast i64 %13 to double
  %27 = bitcast i64 %8 to double
  %28 = load ptr, ptr %0, align 8
  %_M_value.real.i.i = load double, ptr %28, align 8
  %_M_value.imagp.i.i = getelementptr inbounds i8, ptr %28, i64 8
  %_M_value.imag.i.i = load double, ptr %_M_value.imagp.i.i, align 8
  %add.r.i.i = fadd double %_M_value.real.i.i, %27
  %add.i.i.i = fadd double %_M_value.imag.i.i, %26
  store double %add.r.i.i, ptr %28, align 8
  store double %add.i.i.i, ptr %_M_value.imagp.i.i, align 8
  br label %ifcont

ifcont:                                           ; preds = %entry, %then
  %29 = icmp uge i16 %1, %2
  %30 = and i1 %29, %15
  br i1 %30, label %then4, label %ifcont6

then4:                                            ; preds = %ifcont
  %31 = load ptr, ptr %0, align 8
  store i64 %8, ptr %31, align 8, !tbaa.struct !18
  %.omp.reduction.element.sroa.3.0..sroa_idx = getelementptr inbounds i8, ptr %31, i64 8
  store i64 %13, ptr %.omp.reduction.element.sroa.3.0..sroa_idx, align 8, !tbaa !19
  br label %ifcont6

ifcont6:                                          ; preds = %ifcont, %then4
  ret void
}

; Function Attrs: nounwind
declare i32 @__kmpc_get_warp_size() local_unnamed_addr #2

declare i64 @__kmpc_shuffle_int64(i64, i16, i16) local_unnamed_addr

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #5

; Function Attrs: convergent norecurse nounwind
define internal void @_omp_reduction_inter_warp_copy_func(ptr nocapture noundef readonly %0, i32 noundef %1) #6 {
entry:
  %2 = tail call i32 @__kmpc_global_thread_num(ptr addrspacecast (ptr addrspace(1) @1 to ptr)) #2
  %3 = tail call i32 @__kmpc_get_hardware_thread_id_in_block() #2
  %4 = tail call i32 @__kmpc_get_hardware_thread_id_in_block() #2
  %nvptx_lane_id = and i32 %4, 63
  %5 = tail call i32 @__kmpc_get_hardware_thread_id_in_block() #2
  %nvptx_warp_id = ashr i32 %5, 6
  %warp_master = icmp eq i32 %nvptx_lane_id, 0
  %6 = getelementptr inbounds [64 x i32], ptr addrspace(3) @__openmp_nvptx_data_transfer_temporary_storage, i32 0, i32 %nvptx_warp_id
  %is_active_thread = icmp ult i32 %3, %1
  %7 = getelementptr inbounds [64 x i32], ptr addrspace(3) @__openmp_nvptx_data_transfer_temporary_storage, i32 0, i32 %3
  tail call void @__kmpc_barrier(ptr addrspacecast (ptr addrspace(1) @4 to ptr), i32 %2) #2
  br i1 %warp_master, label %then, label %ifcont

then:                                             ; preds = %entry
  %8 = load ptr, ptr %0, align 8, !tbaa !16
  %9 = load i32, ptr %8, align 4
  store volatile i32 %9, ptr addrspace(3) %6, align 4
  br label %ifcont

ifcont:                                           ; preds = %entry, %then
  tail call void @__kmpc_barrier(ptr addrspacecast (ptr addrspace(1) @4 to ptr), i32 %2) #2
  br i1 %is_active_thread, label %then2, label %ifcont4

then2:                                            ; preds = %ifcont
  %10 = load ptr, ptr %0, align 8, !tbaa !16
  %11 = load volatile i32, ptr addrspace(3) %7, align 4, !tbaa !12
  store i32 %11, ptr %10, align 4, !tbaa !12
  br label %ifcont4

ifcont4:                                          ; preds = %ifcont, %then2
  tail call void @__kmpc_barrier(ptr addrspacecast (ptr addrspace(1) @4 to ptr), i32 %2) #2
  br i1 %warp_master, label %then.1, label %ifcont.1

then.1:                                           ; preds = %ifcont4
  %12 = load ptr, ptr %0, align 8, !tbaa !16
  %13 = getelementptr i8, ptr %12, i64 4
  %14 = load i32, ptr %13, align 4
  store volatile i32 %14, ptr addrspace(3) %6, align 4
  br label %ifcont.1

ifcont.1:                                         ; preds = %then.1, %ifcont4
  tail call void @__kmpc_barrier(ptr addrspacecast (ptr addrspace(1) @4 to ptr), i32 %2) #2
  br i1 %is_active_thread, label %then2.1, label %ifcont4.1

then2.1:                                          ; preds = %ifcont.1
  %15 = load ptr, ptr %0, align 8, !tbaa !16
  %16 = getelementptr i8, ptr %15, i64 4
  %17 = load volatile i32, ptr addrspace(3) %7, align 4, !tbaa !12
  store i32 %17, ptr %16, align 4, !tbaa !12
  br label %ifcont4.1

ifcont4.1:                                        ; preds = %then2.1, %ifcont.1
  tail call void @__kmpc_barrier(ptr addrspacecast (ptr addrspace(1) @4 to ptr), i32 %2) #2
  br i1 %warp_master, label %then.2, label %ifcont.2

then.2:                                           ; preds = %ifcont4.1
  %18 = load ptr, ptr %0, align 8, !tbaa !16
  %19 = getelementptr i8, ptr %18, i64 8
  %20 = load i32, ptr %19, align 4
  store volatile i32 %20, ptr addrspace(3) %6, align 4
  br label %ifcont.2

ifcont.2:                                         ; preds = %then.2, %ifcont4.1
  tail call void @__kmpc_barrier(ptr addrspacecast (ptr addrspace(1) @4 to ptr), i32 %2) #2
  br i1 %is_active_thread, label %then2.2, label %ifcont4.2

then2.2:                                          ; preds = %ifcont.2
  %21 = load ptr, ptr %0, align 8, !tbaa !16
  %22 = getelementptr i8, ptr %21, i64 8
  %23 = load volatile i32, ptr addrspace(3) %7, align 4, !tbaa !12
  store i32 %23, ptr %22, align 4, !tbaa !12
  br label %ifcont4.2

ifcont4.2:                                        ; preds = %then2.2, %ifcont.2
  tail call void @__kmpc_barrier(ptr addrspacecast (ptr addrspace(1) @4 to ptr), i32 %2) #2
  br i1 %warp_master, label %then.3, label %ifcont.3

then.3:                                           ; preds = %ifcont4.2
  %24 = load ptr, ptr %0, align 8, !tbaa !16
  %25 = getelementptr i8, ptr %24, i64 12
  %26 = load i32, ptr %25, align 4
  store volatile i32 %26, ptr addrspace(3) %6, align 4
  br label %ifcont.3

ifcont.3:                                         ; preds = %then.3, %ifcont4.2
  tail call void @__kmpc_barrier(ptr addrspacecast (ptr addrspace(1) @4 to ptr), i32 %2) #2
  br i1 %is_active_thread, label %then2.3, label %ifcont4.3

then2.3:                                          ; preds = %ifcont.3
  %27 = load ptr, ptr %0, align 8, !tbaa !16
  %28 = getelementptr i8, ptr %27, i64 12
  %29 = load volatile i32, ptr addrspace(3) %7, align 4, !tbaa !12
  store i32 %29, ptr %28, align 4, !tbaa !12
  br label %ifcont4.3

ifcont4.3:                                        ; preds = %then2.3, %ifcont.3
  ret void
}

; Function Attrs: nounwind
declare i32 @__kmpc_get_hardware_thread_id_in_block() local_unnamed_addr #2

; Function Attrs: nounwind
declare i32 @__kmpc_global_thread_num(ptr) local_unnamed_addr #2

; Function Attrs: convergent nounwind
declare void @__kmpc_barrier(ptr, i32) local_unnamed_addr #7

declare i32 @__kmpc_nvptx_parallel_reduce_nowait_v2(ptr, i64, ptr, ptr, ptr) local_unnamed_addr

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p5(i64 immarg, ptr addrspace(5) nocapture) #1

; Function Attrs: alwaysinline
declare void @__kmpc_parallel_51(ptr, i32, i32, i32, i32, ptr, ptr, ptr, i64) local_unnamed_addr #8

; Function Attrs: nounwind
declare void @__kmpc_distribute_static_fini(ptr, i32) local_unnamed_addr #2

; Function Attrs: norecurse nounwind
define internal void @_omp_reduction_shuffle_and_reduce_func.1(ptr nocapture noundef readonly %0, i16 noundef signext %1, i16 noundef signext %2, i16 noundef signext %3) #4 {
entry:
  %4 = load ptr, ptr %0, align 8
  %5 = load i64, ptr %4, align 8
  %6 = tail call i32 @__kmpc_get_warp_size() #2
  %7 = trunc i32 %6 to i16
  %8 = tail call i64 @__kmpc_shuffle_int64(i64 %5, i16 %2, i16 %7) #2
  %9 = getelementptr i8, ptr %4, i64 8
  %10 = load i64, ptr %9, align 8
  %11 = tail call i32 @__kmpc_get_warp_size() #2
  %12 = trunc i32 %11 to i16
  %13 = tail call i64 @__kmpc_shuffle_int64(i64 %10, i16 %2, i16 %12) #2
  %14 = icmp eq i16 %3, 0
  %15 = icmp eq i16 %3, 1
  %16 = icmp ult i16 %1, %2
  %17 = and i1 %16, %15
  %18 = icmp eq i16 %3, 2
  %19 = and i16 %1, 1
  %20 = icmp eq i16 %19, 0
  %21 = and i1 %20, %18
  %22 = icmp sgt i16 %2, 0
  %23 = and i1 %22, %21
  %24 = or i1 %14, %17
  %25 = or i1 %24, %23
  br i1 %25, label %then, label %ifcont

then:                                             ; preds = %entry
  %26 = bitcast i64 %13 to double
  %27 = bitcast i64 %8 to double
  %28 = load ptr, ptr %0, align 8
  %_M_value.real.i.i = load double, ptr %28, align 8
  %_M_value.imagp.i.i = getelementptr inbounds i8, ptr %28, i64 8
  %_M_value.imag.i.i = load double, ptr %_M_value.imagp.i.i, align 8
  %add.r.i.i = fadd double %_M_value.real.i.i, %27
  %add.i.i.i = fadd double %_M_value.imag.i.i, %26
  store double %add.r.i.i, ptr %28, align 8
  store double %add.i.i.i, ptr %_M_value.imagp.i.i, align 8
  br label %ifcont

ifcont:                                           ; preds = %entry, %then
  %29 = icmp uge i16 %1, %2
  %30 = and i1 %29, %15
  br i1 %30, label %then4, label %ifcont6

then4:                                            ; preds = %ifcont
  %31 = load ptr, ptr %0, align 8
  store i64 %8, ptr %31, align 8, !tbaa.struct !18
  %.omp.reduction.element.sroa.3.0..sroa_idx = getelementptr inbounds i8, ptr %31, i64 8
  store i64 %13, ptr %.omp.reduction.element.sroa.3.0..sroa_idx, align 8, !tbaa !19
  br label %ifcont6

ifcont6:                                          ; preds = %ifcont, %then4
  ret void
}

; Function Attrs: convergent norecurse nounwind
define internal void @_omp_reduction_inter_warp_copy_func.2(ptr nocapture noundef readonly %0, i32 noundef %1) #6 {
entry:
  %2 = tail call i32 @__kmpc_global_thread_num(ptr addrspacecast (ptr addrspace(1) @1 to ptr)) #2
  %3 = tail call i32 @__kmpc_get_hardware_thread_id_in_block() #2
  %4 = tail call i32 @__kmpc_get_hardware_thread_id_in_block() #2
  %nvptx_lane_id = and i32 %4, 63
  %5 = tail call i32 @__kmpc_get_hardware_thread_id_in_block() #2
  %nvptx_warp_id = ashr i32 %5, 6
  %warp_master = icmp eq i32 %nvptx_lane_id, 0
  %6 = getelementptr inbounds [64 x i32], ptr addrspace(3) @__openmp_nvptx_data_transfer_temporary_storage, i32 0, i32 %nvptx_warp_id
  %is_active_thread = icmp ult i32 %3, %1
  %7 = getelementptr inbounds [64 x i32], ptr addrspace(3) @__openmp_nvptx_data_transfer_temporary_storage, i32 0, i32 %3
  tail call void @__kmpc_barrier(ptr addrspacecast (ptr addrspace(1) @4 to ptr), i32 %2) #2
  br i1 %warp_master, label %then, label %ifcont

then:                                             ; preds = %entry
  %8 = load ptr, ptr %0, align 8, !tbaa !16
  %9 = load i32, ptr %8, align 4
  store volatile i32 %9, ptr addrspace(3) %6, align 4
  br label %ifcont

ifcont:                                           ; preds = %entry, %then
  tail call void @__kmpc_barrier(ptr addrspacecast (ptr addrspace(1) @4 to ptr), i32 %2) #2
  br i1 %is_active_thread, label %then2, label %ifcont4

then2:                                            ; preds = %ifcont
  %10 = load ptr, ptr %0, align 8, !tbaa !16
  %11 = load volatile i32, ptr addrspace(3) %7, align 4, !tbaa !12
  store i32 %11, ptr %10, align 4, !tbaa !12
  br label %ifcont4

ifcont4:                                          ; preds = %ifcont, %then2
  tail call void @__kmpc_barrier(ptr addrspacecast (ptr addrspace(1) @4 to ptr), i32 %2) #2
  br i1 %warp_master, label %then.1, label %ifcont.1

then.1:                                           ; preds = %ifcont4
  %12 = load ptr, ptr %0, align 8, !tbaa !16
  %13 = getelementptr i8, ptr %12, i64 4
  %14 = load i32, ptr %13, align 4
  store volatile i32 %14, ptr addrspace(3) %6, align 4
  br label %ifcont.1

ifcont.1:                                         ; preds = %then.1, %ifcont4
  tail call void @__kmpc_barrier(ptr addrspacecast (ptr addrspace(1) @4 to ptr), i32 %2) #2
  br i1 %is_active_thread, label %then2.1, label %ifcont4.1

then2.1:                                          ; preds = %ifcont.1
  %15 = load ptr, ptr %0, align 8, !tbaa !16
  %16 = getelementptr i8, ptr %15, i64 4
  %17 = load volatile i32, ptr addrspace(3) %7, align 4, !tbaa !12
  store i32 %17, ptr %16, align 4, !tbaa !12
  br label %ifcont4.1

ifcont4.1:                                        ; preds = %then2.1, %ifcont.1
  tail call void @__kmpc_barrier(ptr addrspacecast (ptr addrspace(1) @4 to ptr), i32 %2) #2
  br i1 %warp_master, label %then.2, label %ifcont.2

then.2:                                           ; preds = %ifcont4.1
  %18 = load ptr, ptr %0, align 8, !tbaa !16
  %19 = getelementptr i8, ptr %18, i64 8
  %20 = load i32, ptr %19, align 4
  store volatile i32 %20, ptr addrspace(3) %6, align 4
  br label %ifcont.2

ifcont.2:                                         ; preds = %then.2, %ifcont4.1
  tail call void @__kmpc_barrier(ptr addrspacecast (ptr addrspace(1) @4 to ptr), i32 %2) #2
  br i1 %is_active_thread, label %then2.2, label %ifcont4.2

then2.2:                                          ; preds = %ifcont.2
  %21 = load ptr, ptr %0, align 8, !tbaa !16
  %22 = getelementptr i8, ptr %21, i64 8
  %23 = load volatile i32, ptr addrspace(3) %7, align 4, !tbaa !12
  store i32 %23, ptr %22, align 4, !tbaa !12
  br label %ifcont4.2

ifcont4.2:                                        ; preds = %then2.2, %ifcont.2
  tail call void @__kmpc_barrier(ptr addrspacecast (ptr addrspace(1) @4 to ptr), i32 %2) #2
  br i1 %warp_master, label %then.3, label %ifcont.3

then.3:                                           ; preds = %ifcont4.2
  %24 = load ptr, ptr %0, align 8, !tbaa !16
  %25 = getelementptr i8, ptr %24, i64 12
  %26 = load i32, ptr %25, align 4
  store volatile i32 %26, ptr addrspace(3) %6, align 4
  br label %ifcont.3

ifcont.3:                                         ; preds = %then.3, %ifcont4.2
  tail call void @__kmpc_barrier(ptr addrspacecast (ptr addrspace(1) @4 to ptr), i32 %2) #2
  br i1 %is_active_thread, label %then2.3, label %ifcont4.3

then2.3:                                          ; preds = %ifcont.3
  %27 = load ptr, ptr %0, align 8, !tbaa !16
  %28 = getelementptr i8, ptr %27, i64 12
  %29 = load volatile i32, ptr addrspace(3) %7, align 4, !tbaa !12
  store i32 %29, ptr %28, align 4, !tbaa !12
  br label %ifcont4.3

ifcont4.3:                                        ; preds = %then2.3, %ifcont.3
  ret void
}

; Function Attrs: nounwind
declare ptr @__kmpc_reduction_get_fixed_buffer() local_unnamed_addr #2

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none)
define internal void @_omp_reduction_list_to_global_copy_func(ptr nocapture noundef writeonly %0, i32 noundef %1, ptr nocapture noundef readonly %2) #9 {
entry:
  %3 = load ptr, ptr %2, align 8, !tbaa !16
  %4 = sext i32 %1 to i64
  %5 = getelementptr inbounds %struct._globalized_locals_ty, ptr %0, i64 %4
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %5, ptr noundef nonnull align 8 dereferenceable(16) %3, i64 16, i1 false), !tbaa.struct !18
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: readwrite, inaccessiblemem: none)
define internal void @_omp_reduction_list_to_global_reduce_func(ptr nocapture noundef %0, i32 noundef %1, ptr nocapture noundef readonly %2) #10 {
entry:
  %3 = sext i32 %1 to i64
  %4 = getelementptr inbounds %struct._globalized_locals_ty, ptr %0, i64 %3
  %5 = load ptr, ptr %2, align 8
  %_M_value.real.i.i.i = load double, ptr %5, align 8
  %_M_value.imagp.i.i.i = getelementptr inbounds i8, ptr %5, i64 8
  %_M_value.imag.i.i.i = load double, ptr %_M_value.imagp.i.i.i, align 8
  %_M_value.real.i.i = load double, ptr %4, align 8
  %_M_value.imagp.i.i = getelementptr inbounds i8, ptr %4, i64 8
  %_M_value.imag.i.i = load double, ptr %_M_value.imagp.i.i, align 8
  %add.r.i.i = fadd double %_M_value.real.i.i.i, %_M_value.real.i.i
  %add.i.i.i = fadd double %_M_value.imag.i.i.i, %_M_value.imag.i.i
  store double %add.r.i.i, ptr %4, align 8
  store double %add.i.i.i, ptr %_M_value.imagp.i.i, align 8
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none)
define internal void @_omp_reduction_global_to_list_copy_func(ptr nocapture noundef readonly %0, i32 noundef %1, ptr nocapture noundef readonly %2) #9 {
entry:
  %3 = load ptr, ptr %2, align 8, !tbaa !16
  %4 = sext i32 %1 to i64
  %5 = getelementptr inbounds %struct._globalized_locals_ty, ptr %0, i64 %4
  tail call void @llvm.memcpy.p0.p0.i64(ptr noundef nonnull align 8 dereferenceable(16) %3, ptr noundef nonnull align 8 dereferenceable(16) %5, i64 16, i1 false), !tbaa.struct !18
  ret void
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none)
define internal void @_omp_reduction_global_to_list_reduce_func(ptr nocapture noundef readonly %0, i32 noundef %1, ptr nocapture noundef readonly %2) #9 {
entry:
  %3 = sext i32 %1 to i64
  %4 = getelementptr inbounds %struct._globalized_locals_ty, ptr %0, i64 %3
  %5 = load ptr, ptr %2, align 8
  %_M_value.real.i.i.i = load double, ptr %4, align 8
  %_M_value.imagp.i.i.i = getelementptr inbounds i8, ptr %4, i64 8
  %_M_value.imag.i.i.i = load double, ptr %_M_value.imagp.i.i.i, align 8
  %_M_value.real.i.i = load double, ptr %5, align 8
  %_M_value.imagp.i.i = getelementptr inbounds i8, ptr %5, i64 8
  %_M_value.imag.i.i = load double, ptr %_M_value.imagp.i.i, align 8
  %add.r.i.i = fadd double %_M_value.real.i.i.i, %_M_value.real.i.i
  %add.i.i.i = fadd double %_M_value.imag.i.i.i, %_M_value.imag.i.i
  store double %add.r.i.i, ptr %5, align 8
  store double %add.i.i.i, ptr %_M_value.imagp.i.i, align 8
  ret void
}

declare i32 @__kmpc_nvptx_teams_reduce_nowait_v2(ptr, ptr, i32, i64, ptr, ptr, ptr, ptr, ptr, ptr, ptr) local_unnamed_addr

declare void @__kmpc_target_deinit() local_unnamed_addr

; Function Attrs: cold mustprogress noinline nounwind optsize
define weak hidden { double, double } @__muldc3(double noundef %__a, double noundef %__b, double noundef %__c, double noundef %__d) local_unnamed_addr #11 {
entry:
  %mul = fmul double %__a, %__c
  %mul1 = fmul double %__b, %__d
  %mul2 = fmul double %__a, %__d
  %mul3 = fmul double %__b, %__c
  %sub = fsub double %mul, %mul1
  %add = fadd double %mul3, %mul2
  %0 = fcmp ord double %sub, 0.000000e+00
  %1 = fcmp ord double %add, 0.000000e+00
  %or.cond = or i1 %0, %1
  br i1 %or.cond, label %if.end104, label %if.then

if.then:                                          ; preds = %entry
  %2 = tail call double @llvm.fabs.f64(double %__a)
  %3 = fcmp oeq double %2, 0x7FF0000000000000
  %4 = tail call double @llvm.fabs.f64(double %__b)
  %5 = fcmp oeq double %4, 0x7FF0000000000000
  %or.cond158.not = or i1 %3, %5
  br i1 %or.cond158.not, label %if.then12, label %if.end30

if.then12:                                        ; preds = %if.then
  %conv = uitofp i1 %3 to double
  %6 = tail call noundef double @llvm.copysign.f64(double %conv, double %__a)
  %conv19 = uitofp i1 %5 to double
  %7 = tail call noundef double @llvm.copysign.f64(double %conv19, double %__b)
  %8 = fcmp ord double %__c, 0.000000e+00
  %9 = tail call noundef double @llvm.copysign.f64(double 0.000000e+00, double %__c)
  %spec.select = select i1 %8, double %__c, double %9
  %10 = fcmp ord double %__d, 0.000000e+00
  %11 = tail call noundef double @llvm.copysign.f64(double 0.000000e+00, double %__d)
  %spec.select154 = select i1 %10, double %__d, double %11
  br label %if.end30

if.end30:                                         ; preds = %if.then, %if.then12
  %__d.addr.1 = phi double [ %spec.select154, %if.then12 ], [ %__d, %if.then ]
  %__c.addr.1 = phi double [ %spec.select, %if.then12 ], [ %__c, %if.then ]
  %__b.addr.0 = phi double [ %7, %if.then12 ], [ %__b, %if.then ]
  %__a.addr.0 = phi double [ %6, %if.then12 ], [ %__a, %if.then ]
  %__recalc.0 = phi i32 [ 1, %if.then12 ], [ 0, %if.then ]
  %12 = tail call double @llvm.fabs.f64(double %__c.addr.1)
  %13 = fcmp oeq double %12, 0x7FF0000000000000
  %14 = tail call double @llvm.fabs.f64(double %__d.addr.1)
  %15 = fcmp oeq double %14, 0x7FF0000000000000
  %or.cond161.not = or i1 %15, %13
  br i1 %or.cond161.not, label %if.then36, label %if.end57

if.then36:                                        ; preds = %if.end30
  %conv40 = uitofp i1 %13 to double
  %16 = tail call noundef double @llvm.copysign.f64(double %conv40, double %__c.addr.1)
  %conv45 = uitofp i1 %15 to double
  %17 = tail call noundef double @llvm.copysign.f64(double %conv45, double %__d.addr.1)
  %18 = fcmp ord double %__a.addr.0, 0.000000e+00
  %19 = tail call noundef double @llvm.copysign.f64(double 0.000000e+00, double %__a.addr.0)
  %spec.select152 = select i1 %18, double %__a.addr.0, double %19
  %20 = fcmp ord double %__b.addr.0, 0.000000e+00
  %21 = tail call noundef double @llvm.copysign.f64(double 0.000000e+00, double %__b.addr.0)
  %spec.select155 = select i1 %20, double %__b.addr.0, double %21
  br label %if.end57

if.end57:                                         ; preds = %if.end30, %if.then36
  %__d.addr.2 = phi double [ %17, %if.then36 ], [ %__d.addr.1, %if.end30 ]
  %__c.addr.2 = phi double [ %16, %if.then36 ], [ %__c.addr.1, %if.end30 ]
  %__b.addr.2 = phi double [ %spec.select155, %if.then36 ], [ %__b.addr.0, %if.end30 ]
  %__a.addr.2 = phi double [ %spec.select152, %if.then36 ], [ %__a.addr.0, %if.end30 ]
  %__recalc.1 = phi i32 [ 1, %if.then36 ], [ %__recalc.0, %if.end30 ]
  %tobool58.not = icmp eq i32 %__recalc.1, 0
  br i1 %tobool58.not, label %land.lhs.true59, label %if.end92

land.lhs.true59:                                  ; preds = %if.end57
  %22 = tail call double @llvm.fabs.f64(double %mul)
  %23 = fcmp une double %22, 0x7FF0000000000000
  %24 = tail call double @llvm.fabs.f64(double %mul1)
  %25 = fcmp une double %24, 0x7FF0000000000000
  %or.cond163 = and i1 %23, %25
  %26 = tail call double @llvm.fabs.f64(double %mul2)
  %27 = fcmp une double %26, 0x7FF0000000000000
  %or.cond165 = and i1 %27, %or.cond163
  %28 = tail call double @llvm.fabs.f64(double %mul3)
  %29 = fcmp une double %28, 0x7FF0000000000000
  %or.cond167 = and i1 %29, %or.cond165
  br i1 %or.cond167, label %if.end92, label %if.then71

if.then71:                                        ; preds = %land.lhs.true59
  %30 = fcmp ord double %__a.addr.2, 0.000000e+00
  %31 = tail call noundef double @llvm.copysign.f64(double 0.000000e+00, double %__a.addr.2)
  %spec.select153 = select i1 %30, double %__a.addr.2, double %31
  %32 = fcmp ord double %__b.addr.2, 0.000000e+00
  %33 = tail call noundef double @llvm.copysign.f64(double 0.000000e+00, double %__b.addr.2)
  %__b.addr.3 = select i1 %32, double %__b.addr.2, double %33
  %34 = fcmp ord double %__c.addr.2, 0.000000e+00
  %35 = tail call noundef double @llvm.copysign.f64(double 0.000000e+00, double %__c.addr.2)
  %__c.addr.3 = select i1 %34, double %__c.addr.2, double %35
  %36 = fcmp ord double %__d.addr.2, 0.000000e+00
  %37 = tail call noundef double @llvm.copysign.f64(double 0.000000e+00, double %__d.addr.2)
  %spec.select156 = select i1 %36, double %__d.addr.2, double %37
  br label %if.end92

if.end92:                                         ; preds = %land.lhs.true59, %if.then71, %if.end57
  %__d.addr.4 = phi double [ %__d.addr.2, %if.end57 ], [ %spec.select156, %if.then71 ], [ %__d.addr.2, %land.lhs.true59 ]
  %__c.addr.4 = phi double [ %__c.addr.2, %if.end57 ], [ %__c.addr.3, %if.then71 ], [ %__c.addr.2, %land.lhs.true59 ]
  %__b.addr.4 = phi double [ %__b.addr.2, %if.end57 ], [ %__b.addr.3, %if.then71 ], [ %__b.addr.2, %land.lhs.true59 ]
  %__a.addr.4 = phi double [ %__a.addr.2, %if.end57 ], [ %spec.select153, %if.then71 ], [ %__a.addr.2, %land.lhs.true59 ]
  %tobool93.not = phi i1 [ false, %if.end57 ], [ false, %if.then71 ], [ true, %land.lhs.true59 ]
  br i1 %tobool93.not, label %if.end104, label %if.then94

if.then94:                                        ; preds = %if.end92
  %38 = fneg double %__b.addr.4
  %neg = fmul double %__d.addr.4, %38
  %39 = tail call double @llvm.fmuladd.f64(double %__a.addr.4, double %__c.addr.4, double %neg)
  %mul97 = fmul double %39, 0x7FF0000000000000
  %mul100 = fmul double %__c.addr.4, %__b.addr.4
  %40 = tail call double @llvm.fmuladd.f64(double %__a.addr.4, double %__d.addr.4, double %mul100)
  %mul101 = fmul double %40, 0x7FF0000000000000
  br label %if.end104

if.end104:                                        ; preds = %if.end92, %if.then94, %entry
  %z.sroa.6.1 = phi double [ %add, %entry ], [ %mul101, %if.then94 ], [ %add, %if.end92 ]
  %z.sroa.0.1 = phi double [ %sub, %entry ], [ %mul97, %if.then94 ], [ %sub, %if.end92 ]
  %.fca.0.insert = insertvalue { double, double } poison, double %z.sroa.0.1, 0
  %.fca.1.insert = insertvalue { double, double } %.fca.0.insert, double %z.sroa.6.1, 1
  ret { double, double } %.fca.1.insert
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #12

; Function Attrs: cold mustprogress noinline nounwind optsize
define weak hidden [2 x i32] @__mulsc3(float noundef %__a, float noundef %__b, float noundef %__c, float noundef %__d) local_unnamed_addr #11 {
entry:
  %mul = fmul float %__a, %__c
  %mul1 = fmul float %__b, %__d
  %mul2 = fmul float %__a, %__d
  %mul3 = fmul float %__b, %__c
  %sub = fsub float %mul, %mul1
  %add = fadd float %mul3, %mul2
  %0 = fcmp ord float %sub, 0.000000e+00
  %1 = fcmp ord float %add, 0.000000e+00
  %or.cond = or i1 %0, %1
  br i1 %or.cond, label %if.end104, label %if.then

if.then:                                          ; preds = %entry
  %2 = tail call float @llvm.fabs.f32(float %__a)
  %3 = fcmp oeq float %2, 0x7FF0000000000000
  %4 = tail call float @llvm.fabs.f32(float %__b)
  %5 = fcmp oeq float %4, 0x7FF0000000000000
  %or.cond160.not = or i1 %3, %5
  br i1 %or.cond160.not, label %if.then12, label %if.end30

if.then12:                                        ; preds = %if.then
  %conv = uitofp i1 %3 to float
  %6 = tail call noundef float @llvm.copysign.f32(float %conv, float %__a)
  %conv19 = uitofp i1 %5 to float
  %7 = tail call noundef float @llvm.copysign.f32(float %conv19, float %__b)
  %8 = fcmp ord float %__c, 0.000000e+00
  %9 = tail call noundef float @llvm.copysign.f32(float 0.000000e+00, float %__c)
  %spec.select = select i1 %8, float %__c, float %9
  %10 = fcmp ord float %__d, 0.000000e+00
  %11 = tail call noundef float @llvm.copysign.f32(float 0.000000e+00, float %__d)
  %spec.select156 = select i1 %10, float %__d, float %11
  br label %if.end30

if.end30:                                         ; preds = %if.then, %if.then12
  %__d.addr.1 = phi float [ %spec.select156, %if.then12 ], [ %__d, %if.then ]
  %__c.addr.1 = phi float [ %spec.select, %if.then12 ], [ %__c, %if.then ]
  %__b.addr.0 = phi float [ %7, %if.then12 ], [ %__b, %if.then ]
  %__a.addr.0 = phi float [ %6, %if.then12 ], [ %__a, %if.then ]
  %__recalc.0 = phi i32 [ 1, %if.then12 ], [ 0, %if.then ]
  %12 = tail call float @llvm.fabs.f32(float %__c.addr.1)
  %13 = fcmp oeq float %12, 0x7FF0000000000000
  %14 = tail call float @llvm.fabs.f32(float %__d.addr.1)
  %15 = fcmp oeq float %14, 0x7FF0000000000000
  %or.cond163.not = or i1 %15, %13
  br i1 %or.cond163.not, label %if.then36, label %if.end57

if.then36:                                        ; preds = %if.end30
  %conv40 = uitofp i1 %13 to float
  %16 = tail call noundef float @llvm.copysign.f32(float %conv40, float %__c.addr.1)
  %conv45 = uitofp i1 %15 to float
  %17 = tail call noundef float @llvm.copysign.f32(float %conv45, float %__d.addr.1)
  %18 = fcmp ord float %__a.addr.0, 0.000000e+00
  %19 = tail call noundef float @llvm.copysign.f32(float 0.000000e+00, float %__a.addr.0)
  %spec.select152 = select i1 %18, float %__a.addr.0, float %19
  %20 = fcmp ord float %__b.addr.0, 0.000000e+00
  %21 = tail call noundef float @llvm.copysign.f32(float 0.000000e+00, float %__b.addr.0)
  %spec.select157 = select i1 %20, float %__b.addr.0, float %21
  br label %if.end57

if.end57:                                         ; preds = %if.end30, %if.then36
  %__d.addr.2 = phi float [ %17, %if.then36 ], [ %__d.addr.1, %if.end30 ]
  %__c.addr.2 = phi float [ %16, %if.then36 ], [ %__c.addr.1, %if.end30 ]
  %__b.addr.2 = phi float [ %spec.select157, %if.then36 ], [ %__b.addr.0, %if.end30 ]
  %__a.addr.2 = phi float [ %spec.select152, %if.then36 ], [ %__a.addr.0, %if.end30 ]
  %__recalc.1 = phi i32 [ 1, %if.then36 ], [ %__recalc.0, %if.end30 ]
  %tobool58.not = icmp eq i32 %__recalc.1, 0
  br i1 %tobool58.not, label %land.lhs.true59, label %if.end92

land.lhs.true59:                                  ; preds = %if.end57
  %22 = tail call float @llvm.fabs.f32(float %mul)
  %23 = fcmp une float %22, 0x7FF0000000000000
  %24 = tail call float @llvm.fabs.f32(float %mul1)
  %25 = fcmp une float %24, 0x7FF0000000000000
  %or.cond165 = and i1 %23, %25
  %26 = tail call float @llvm.fabs.f32(float %mul2)
  %27 = fcmp une float %26, 0x7FF0000000000000
  %or.cond167 = and i1 %27, %or.cond165
  %28 = tail call float @llvm.fabs.f32(float %mul3)
  %29 = fcmp une float %28, 0x7FF0000000000000
  %or.cond169 = and i1 %29, %or.cond167
  br i1 %or.cond169, label %if.end92, label %if.then71

if.then71:                                        ; preds = %land.lhs.true59
  %30 = fcmp ord float %__a.addr.2, 0.000000e+00
  %31 = tail call noundef float @llvm.copysign.f32(float 0.000000e+00, float %__a.addr.2)
  %spec.select153 = select i1 %30, float %__a.addr.2, float %31
  %32 = fcmp ord float %__b.addr.2, 0.000000e+00
  %33 = tail call noundef float @llvm.copysign.f32(float 0.000000e+00, float %__b.addr.2)
  %__b.addr.3 = select i1 %32, float %__b.addr.2, float %33
  %34 = fcmp ord float %__c.addr.2, 0.000000e+00
  %35 = tail call noundef float @llvm.copysign.f32(float 0.000000e+00, float %__c.addr.2)
  %__c.addr.3 = select i1 %34, float %__c.addr.2, float %35
  %36 = fcmp ord float %__d.addr.2, 0.000000e+00
  %37 = tail call noundef float @llvm.copysign.f32(float 0.000000e+00, float %__d.addr.2)
  %spec.select158 = select i1 %36, float %__d.addr.2, float %37
  br label %if.end92

if.end92:                                         ; preds = %land.lhs.true59, %if.then71, %if.end57
  %__d.addr.4 = phi float [ %__d.addr.2, %if.end57 ], [ %spec.select158, %if.then71 ], [ %__d.addr.2, %land.lhs.true59 ]
  %__c.addr.4 = phi float [ %__c.addr.2, %if.end57 ], [ %__c.addr.3, %if.then71 ], [ %__c.addr.2, %land.lhs.true59 ]
  %__b.addr.4 = phi float [ %__b.addr.2, %if.end57 ], [ %__b.addr.3, %if.then71 ], [ %__b.addr.2, %land.lhs.true59 ]
  %__a.addr.4 = phi float [ %__a.addr.2, %if.end57 ], [ %spec.select153, %if.then71 ], [ %__a.addr.2, %land.lhs.true59 ]
  %tobool93.not = phi i1 [ false, %if.end57 ], [ false, %if.then71 ], [ true, %land.lhs.true59 ]
  %38 = fneg float %__b.addr.4
  %neg = fmul float %__d.addr.4, %38
  %39 = tail call float @llvm.fmuladd.f32(float %__a.addr.4, float %__c.addr.4, float %neg)
  %mul97 = fmul float %39, 0x7FF0000000000000
  %mul100 = fmul float %__c.addr.4, %__b.addr.4
  %40 = tail call float @llvm.fmuladd.f32(float %__a.addr.4, float %__d.addr.4, float %mul100)
  %mul101 = fmul float %40, 0x7FF0000000000000
  %spec.select154 = select i1 %tobool93.not, float %add, float %mul101
  %spec.select155 = select i1 %tobool93.not, float %sub, float %mul97
  br label %if.end104

if.end104:                                        ; preds = %if.end92, %entry
  %z.sroa.6.1 = phi float [ %add, %entry ], [ %spec.select154, %if.end92 ]
  %z.sroa.0.1 = phi float [ %sub, %entry ], [ %spec.select155, %if.end92 ]
  %41 = bitcast float %z.sroa.0.1 to i32
  %.fca.0.insert = insertvalue [2 x i32] poison, i32 %41, 0
  %42 = bitcast float %z.sroa.6.1 to i32
  %.fca.1.insert = insertvalue [2 x i32] %.fca.0.insert, i32 %42, 1
  ret [2 x i32] %.fca.1.insert
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fmuladd.f32(float, float, float) #12

; Function Attrs: cold mustprogress noinline nounwind optsize
define weak hidden { double, double } @__divdc3(double noundef %__a, double noundef %__b, double noundef %__c, double noundef %__d) local_unnamed_addr #11 {
entry:
  %0 = tail call noundef double @llvm.fabs.f64(double %__c)
  %1 = tail call noundef double @llvm.fabs.f64(double %__d)
  %2 = tail call noundef double @llvm.maxnum.f64(double %0, double %1)
  %3 = tail call { double, i32 } @llvm.frexp.f64.i32(double %2)
  %4 = extractvalue { double, i32 } %3, 1
  %5 = add nsw i32 %4, -1
  %6 = sitofp i32 %5 to double
  %7 = fcmp one double %2, 0x7FF0000000000000
  %8 = select i1 %7, double %6, double %2
  %9 = fcmp oeq double %2, 0.000000e+00
  %10 = select i1 %9, double 0xFFF0000000000000, double %8
  %11 = tail call double @llvm.fabs.f64(double %10)
  %12 = fcmp ueq double %11, 0x7FF0000000000000
  %conv = fptosi double %10 to i32
  %sub = sub nsw i32 0, %conv
  %13 = tail call noundef double @llvm.ldexp.f64.i32(double %__c, i32 %sub)
  %14 = tail call noundef double @llvm.ldexp.f64.i32(double %__d, i32 %sub)
  %__c.addr.0 = select i1 %12, double %__c, double %13
  %__d.addr.0 = select i1 %12, double %__d, double %14
  %__ilogbw.0 = select i1 %12, i32 0, i32 %conv
  %mul8 = fmul double %__d.addr.0, %__d.addr.0
  %15 = tail call double @llvm.fmuladd.f64(double %__c.addr.0, double %__c.addr.0, double %mul8)
  %mul9 = fmul double %__d.addr.0, %__b
  %16 = tail call double @llvm.fmuladd.f64(double %__a, double %__c.addr.0, double %mul9)
  %div = fdiv double %16, %15
  %sub10 = sub nsw i32 0, %__ilogbw.0
  %17 = tail call noundef double @llvm.ldexp.f64.i32(double %div, i32 %sub10)
  %18 = fneg double %__d.addr.0
  %neg = fmul double %18, %__a
  %19 = tail call double @llvm.fmuladd.f64(double %__b, double %__c.addr.0, double %neg)
  %div13 = fdiv double %19, %15
  %20 = tail call noundef double @llvm.ldexp.f64.i32(double %div13, i32 %sub10)
  %21 = fcmp ord double %17, 0.000000e+00
  %22 = fcmp ord double %20, 0.000000e+00
  %or.cond153 = or i1 %21, %22
  br i1 %or.cond153, label %if.end94, label %if.then22

if.then22:                                        ; preds = %entry
  %cmp = fcmp oeq double %15, 0.000000e+00
  br i1 %cmp, label %land.lhs.true23, label %if.else

land.lhs.true23:                                  ; preds = %if.then22
  %23 = fcmp ord double %__a, 0.000000e+00
  %24 = fcmp ord double %__b, 0.000000e+00
  %or.cond154 = or i1 %23, %24
  br i1 %or.cond154, label %if.then28, label %if.else

if.then28:                                        ; preds = %land.lhs.true23
  %25 = tail call noundef double @llvm.copysign.f64(double 0x7FF0000000000000, double %__c.addr.0)
  %mul = fmul double %25, %__a
  %mul32 = fmul double %25, %__b
  br label %if.end94

if.else:                                          ; preds = %land.lhs.true23, %if.then22
  %26 = tail call double @llvm.fabs.f64(double %__a)
  %27 = fcmp une double %26, 0x7FF0000000000000
  %28 = tail call double @llvm.fabs.f64(double %__b)
  %29 = fcmp une double %28, 0x7FF0000000000000
  %or.cond156 = and i1 %27, %29
  %30 = tail call double @llvm.fabs.f64(double %__c.addr.0)
  %31 = fcmp ueq double %30, 0x7FF0000000000000
  %or.cond158 = select i1 %or.cond156, i1 true, i1 %31
  %32 = tail call double @llvm.fabs.f64(double %__d.addr.0)
  %33 = fcmp ueq double %32, 0x7FF0000000000000
  %or.cond160 = select i1 %or.cond158, i1 true, i1 %33
  br i1 %or.cond160, label %if.else62, label %if.then45

if.then45:                                        ; preds = %if.else
  %cond = select i1 %27, double 0.000000e+00, double 1.000000e+00
  %34 = tail call noundef double @llvm.copysign.f64(double %cond, double %__a)
  %cond51 = select i1 %29, double 0.000000e+00, double 1.000000e+00
  %35 = tail call noundef double @llvm.copysign.f64(double %cond51, double %__b)
  %mul54 = fmul double %35, %__d.addr.0
  %36 = tail call double @llvm.fmuladd.f64(double %34, double %__c.addr.0, double %mul54)
  %mul55 = fmul double %36, 0x7FF0000000000000
  %37 = fneg double %34
  %neg59 = fmul double %__d.addr.0, %37
  %38 = tail call double @llvm.fmuladd.f64(double %35, double %__c.addr.0, double %neg59)
  %mul60 = fmul double %38, 0x7FF0000000000000
  br label %if.end94

if.else62:                                        ; preds = %if.else
  %or.cond = fcmp une double %10, 0x7FF0000000000000
  %39 = fcmp ueq double %26, 0x7FF0000000000000
  %or.cond161 = or i1 %39, %or.cond
  %40 = fcmp ueq double %28, 0x7FF0000000000000
  %or.cond163 = or i1 %40, %or.cond161
  br i1 %or.cond163, label %if.end94, label %if.then73

if.then73:                                        ; preds = %if.else62
  %41 = fcmp une double %30, 0x7FF0000000000000
  %cond76 = select i1 %41, double 0.000000e+00, double 1.000000e+00
  %42 = tail call noundef double @llvm.copysign.f64(double %cond76, double %__c.addr.0)
  %43 = fcmp une double %32, 0x7FF0000000000000
  %cond80 = select i1 %43, double 0.000000e+00, double 1.000000e+00
  %44 = tail call noundef double @llvm.copysign.f64(double %cond80, double %__d.addr.0)
  %mul83 = fmul double %44, %__b
  %45 = tail call double @llvm.fmuladd.f64(double %__a, double %42, double %mul83)
  %mul84 = fmul double %45, 0.000000e+00
  %46 = fneg double %44
  %neg88 = fmul double %46, %__a
  %47 = tail call double @llvm.fmuladd.f64(double %__b, double %42, double %neg88)
  %mul89 = fmul double %47, 0.000000e+00
  br label %if.end94

if.end94:                                         ; preds = %if.then28, %if.else62, %if.then73, %if.then45, %entry
  %z.sroa.8.0 = phi double [ %mul60, %if.then45 ], [ %mul89, %if.then73 ], [ %20, %if.else62 ], [ %mul32, %if.then28 ], [ %20, %entry ]
  %z.sroa.0.0 = phi double [ %mul55, %if.then45 ], [ %mul84, %if.then73 ], [ %17, %if.else62 ], [ %mul, %if.then28 ], [ %17, %entry ]
  %.fca.0.insert = insertvalue { double, double } poison, double %z.sroa.0.0, 0
  %.fca.1.insert = insertvalue { double, double } %.fca.0.insert, double %z.sroa.8.0, 1
  ret { double, double } %.fca.1.insert
}

; Function Attrs: cold mustprogress noinline nounwind optsize
define weak hidden [2 x i32] @__divsc3(float noundef %__a, float noundef %__b, float noundef %__c, float noundef %__d) local_unnamed_addr #11 {
entry:
  %0 = tail call noundef float @llvm.fabs.f32(float %__c)
  %1 = tail call noundef float @llvm.fabs.f32(float %__d)
  %2 = tail call noundef float @llvm.maxnum.f32(float %0, float %1)
  %3 = fpext float %2 to double
  %4 = tail call { double, i32 } @llvm.frexp.f64.i32(double %3)
  %5 = extractvalue { double, i32 } %4, 1
  %6 = add nsw i32 %5, -1
  %7 = sitofp i32 %6 to float
  %8 = fcmp one float %2, 0x7FF0000000000000
  %9 = select i1 %8, float %7, float %2
  %10 = fcmp oeq float %2, 0.000000e+00
  %11 = select i1 %10, float 0xFFF0000000000000, float %9
  %12 = tail call float @llvm.fabs.f32(float %11)
  %13 = fcmp ueq float %12, 0x7FF0000000000000
  %conv = fptosi float %11 to i32
  %sub = sub nsw i32 0, %conv
  %14 = tail call noundef float @llvm.ldexp.f32.i32(float %__c, i32 %sub)
  %15 = tail call noundef float @llvm.ldexp.f32.i32(float %__d, i32 %sub)
  %__c.addr.0 = select i1 %13, float %__c, float %14
  %__d.addr.0 = select i1 %13, float %__d, float %15
  %__ilogbw.0 = select i1 %13, i32 0, i32 %conv
  %mul8 = fmul float %__d.addr.0, %__d.addr.0
  %16 = tail call float @llvm.fmuladd.f32(float %__c.addr.0, float %__c.addr.0, float %mul8)
  %mul9 = fmul float %__d.addr.0, %__b
  %17 = tail call float @llvm.fmuladd.f32(float %__a, float %__c.addr.0, float %mul9)
  %div = fdiv float %17, %16
  %sub10 = sub nsw i32 0, %__ilogbw.0
  %18 = tail call noundef float @llvm.ldexp.f32.i32(float %div, i32 %sub10)
  %19 = fneg float %__d.addr.0
  %neg = fmul float %19, %__a
  %20 = tail call float @llvm.fmuladd.f32(float %__b, float %__c.addr.0, float %neg)
  %div13 = fdiv float %20, %16
  %21 = tail call noundef float @llvm.ldexp.f32.i32(float %div13, i32 %sub10)
  %22 = fcmp ord float %18, 0.000000e+00
  %23 = fcmp ord float %21, 0.000000e+00
  %or.cond157 = or i1 %22, %23
  br i1 %or.cond157, label %if.end98, label %if.then22

if.then22:                                        ; preds = %entry
  %cmp = fcmp oeq float %16, 0.000000e+00
  br i1 %cmp, label %land.lhs.true23, label %if.else

land.lhs.true23:                                  ; preds = %if.then22
  %24 = fcmp ord float %__a, 0.000000e+00
  %25 = fcmp ord float %__b, 0.000000e+00
  %or.cond158 = or i1 %24, %25
  br i1 %or.cond158, label %if.then28, label %if.else

if.then28:                                        ; preds = %land.lhs.true23
  %26 = tail call noundef float @llvm.copysign.f32(float 0x7FF0000000000000, float %__c.addr.0)
  %mul = fmul float %26, %__a
  %mul32 = fmul float %26, %__b
  br label %if.end98

if.else:                                          ; preds = %land.lhs.true23, %if.then22
  %27 = tail call float @llvm.fabs.f32(float %__a)
  %28 = fcmp oeq float %27, 0x7FF0000000000000
  %.not = xor i1 %28, true
  %29 = tail call float @llvm.fabs.f32(float %__b)
  %30 = fcmp une float %29, 0x7FF0000000000000
  %or.cond160 = and i1 %30, %.not
  %31 = tail call float @llvm.fabs.f32(float %__c.addr.0)
  %32 = fcmp ueq float %31, 0x7FF0000000000000
  %or.cond162 = select i1 %or.cond160, i1 true, i1 %32
  %33 = tail call float @llvm.fabs.f32(float %__d.addr.0)
  %34 = fcmp ueq float %33, 0x7FF0000000000000
  %or.cond164 = select i1 %or.cond162, i1 true, i1 %34
  br i1 %or.cond164, label %if.else64, label %if.then45

if.then45:                                        ; preds = %if.else
  %conv48 = uitofp i1 %28 to float
  %35 = tail call noundef float @llvm.copysign.f32(float %conv48, float %__a)
  %36 = fcmp oeq float %29, 0x7FF0000000000000
  %conv53 = uitofp i1 %36 to float
  %37 = tail call noundef float @llvm.copysign.f32(float %conv53, float %__b)
  %mul56 = fmul float %37, %__d.addr.0
  %38 = tail call float @llvm.fmuladd.f32(float %35, float %__c.addr.0, float %mul56)
  %mul57 = fmul float %38, 0x7FF0000000000000
  %39 = fneg float %35
  %neg61 = fmul float %__d.addr.0, %39
  %40 = tail call float @llvm.fmuladd.f32(float %37, float %__c.addr.0, float %neg61)
  %mul62 = fmul float %40, 0x7FF0000000000000
  br label %if.end98

if.else64:                                        ; preds = %if.else
  %or.cond = fcmp une float %11, 0x7FF0000000000000
  %41 = fcmp ueq float %27, 0x7FF0000000000000
  %or.cond165 = or i1 %41, %or.cond
  %42 = fcmp ueq float %29, 0x7FF0000000000000
  %or.cond167 = or i1 %42, %or.cond165
  br i1 %or.cond167, label %if.end98, label %if.then75

if.then75:                                        ; preds = %if.else64
  %43 = fcmp oeq float %31, 0x7FF0000000000000
  %conv79 = uitofp i1 %43 to float
  %44 = tail call noundef float @llvm.copysign.f32(float %conv79, float %__c.addr.0)
  %45 = fcmp oeq float %33, 0x7FF0000000000000
  %conv84 = uitofp i1 %45 to float
  %46 = tail call noundef float @llvm.copysign.f32(float %conv84, float %__d.addr.0)
  %mul87 = fmul float %46, %__b
  %47 = tail call float @llvm.fmuladd.f32(float %__a, float %44, float %mul87)
  %mul88 = fmul float %47, 0.000000e+00
  %48 = fneg float %46
  %neg92 = fmul float %48, %__a
  %49 = tail call float @llvm.fmuladd.f32(float %__b, float %44, float %neg92)
  %mul93 = fmul float %49, 0.000000e+00
  br label %if.end98

if.end98:                                         ; preds = %if.then28, %if.else64, %if.then75, %if.then45, %entry
  %z.sroa.8.0 = phi float [ %mul62, %if.then45 ], [ %mul93, %if.then75 ], [ %21, %if.else64 ], [ %mul32, %if.then28 ], [ %21, %entry ]
  %z.sroa.0.0 = phi float [ %mul57, %if.then45 ], [ %mul88, %if.then75 ], [ %18, %if.else64 ], [ %mul, %if.then28 ], [ %18, %entry ]
  %50 = bitcast float %z.sroa.0.0 to i32
  %.fca.0.insert = insertvalue [2 x i32] poison, i32 %50, 0
  %51 = bitcast float %z.sroa.8.0 to i32
  %.fca.1.insert = insertvalue [2 x i32] %.fca.0.insert, i32 %51, 1
  ret [2 x i32] %.fca.1.insert
}

; Function Attrs: cold mustprogress noinline nounwind optsize
define weak hidden { double, double } @cexp(double noundef %_a.coerce0, double noundef %_a.coerce1) local_unnamed_addr #11 {
entry:
  %0 = tail call double @llvm.fabs.f64(double %_a.coerce1) #14
  %1 = fcmp olt double %0, 0x41D0000000000000
  br i1 %1, label %2, label %21

2:                                                ; preds = %entry
  %3 = fmul double %0, 0x3FE45F306DC9C883
  %4 = tail call double @llvm.rint.f64(double %3)
  %5 = tail call double @llvm.fma.f64(double %4, double 0xBFF921FB54442D18, double %0)
  %6 = tail call double @llvm.fma.f64(double %4, double 0xBC91A62633145C00, double %5)
  %7 = fmul double %4, 0x3C91A62633145C00
  %8 = fneg double %7
  %9 = tail call double @llvm.fma.f64(double %4, double 0x3C91A62633145C00, double %8)
  %10 = fsub double %5, %7
  %11 = fsub double %5, %10
  %12 = fsub double %11, %7
  %13 = fsub double %10, %6
  %14 = fadd double %13, %12
  %15 = fsub double %14, %9
  %16 = tail call double @llvm.fma.f64(double %4, double 0xB97B839A252049C0, double %15)
  %17 = fadd double %6, %16
  %18 = fsub double %17, %6
  %19 = fsub double %16, %18
  %20 = fptosi double %4 to i32
  br label %__ocml_cexp_f64.exit

21:                                               ; preds = %entry
  %22 = tail call double @llvm.amdgcn.trig.preop.f64(double %0, i32 0)
  %23 = tail call double @llvm.amdgcn.trig.preop.f64(double %0, i32 1)
  %24 = fcmp oge double %0, 0x7B00000000000000
  %25 = tail call double @llvm.ldexp.f64.i32(double %0, i32 -128)
  %26 = select i1 %24, double %25, double %0
  %27 = fmul double %23, %26
  %28 = fmul double %22, %26
  %29 = fneg double %28
  %30 = tail call double @llvm.fma.f64(double %22, double %26, double %29)
  %31 = fadd double %27, %30
  %32 = fadd double %28, %31
  %33 = tail call double @llvm.ldexp.f64.i32(double %32, i32 -2)
  %34 = tail call double @llvm.floor.f64(double %33)
  %35 = fsub double %33, %34
  %36 = tail call double @llvm.minnum.f64(double %35, double 0x3FEFFFFFFFFFFFFF)
  %37 = fcmp uno double %33, 0.000000e+00
  %38 = select i1 %37, double %33, double %36
  %39 = tail call double @llvm.fabs.f64(double %33)
  %40 = fcmp oeq double %39, 0x7FF0000000000000
  %41 = select i1 %40, double 0.000000e+00, double %38
  %42 = fsub double %31, %27
  %43 = fsub double %30, %42
  %44 = fsub double %31, %42
  %45 = fsub double %27, %44
  %46 = fadd double %43, %45
  %47 = fneg double %27
  %48 = tail call double @llvm.fma.f64(double %23, double %26, double %47)
  %49 = tail call double @llvm.amdgcn.trig.preop.f64(double %0, i32 2)
  %50 = fmul double %49, %26
  %51 = fadd double %50, %48
  %52 = fadd double %51, %46
  %53 = fsub double %32, %28
  %54 = fsub double %31, %53
  %55 = fadd double %54, %52
  %56 = fsub double %55, %54
  %57 = fsub double %52, %56
  %58 = fsub double %52, %51
  %59 = fsub double %46, %58
  %60 = fsub double %52, %58
  %61 = fsub double %51, %60
  %62 = fadd double %59, %61
  %63 = fsub double %51, %50
  %64 = fsub double %48, %63
  %65 = fsub double %51, %63
  %66 = fsub double %50, %65
  %67 = fadd double %64, %66
  %68 = fadd double %67, %62
  %69 = fneg double %50
  %70 = tail call double @llvm.fma.f64(double %49, double %26, double %69)
  %71 = fadd double %70, %68
  %72 = fadd double %57, %71
  %73 = tail call double @llvm.ldexp.f64.i32(double %41, i32 2)
  %74 = fadd double %55, %73
  %75 = fcmp olt double %74, 0.000000e+00
  %76 = select i1 %75, double 4.000000e+00, double 0.000000e+00
  %77 = fadd double %73, %76
  %78 = fadd double %55, %77
  %79 = fptosi double %78 to i32
  %80 = sitofp i32 %79 to double
  %81 = fsub double %77, %80
  %82 = fadd double %55, %81
  %83 = fsub double %82, %81
  %84 = fsub double %55, %83
  %85 = fadd double %72, %84
  %86 = fcmp oge double %82, 5.000000e-01
  %87 = zext i1 %86 to i32
  %88 = add nsw i32 %87, %79
  %89 = select i1 %86, double 1.000000e+00, double 0.000000e+00
  %90 = fsub double %82, %89
  %91 = fadd double %90, %85
  %92 = fsub double %91, %90
  %93 = fsub double %85, %92
  %94 = fmul double %91, 0x3FF921FB54442D18
  %95 = fneg double %94
  %96 = tail call double @llvm.fma.f64(double %91, double 0x3FF921FB54442D18, double %95)
  %97 = tail call double @llvm.fma.f64(double %91, double 0x3C91A62633145C07, double %96)
  %98 = tail call double @llvm.fma.f64(double %93, double 0x3FF921FB54442D18, double %97)
  %99 = fadd double %94, %98
  %100 = fsub double %99, %94
  %101 = fsub double %98, %100
  br label %__ocml_cexp_f64.exit

__ocml_cexp_f64.exit:                             ; preds = %2, %21
  %.pn5.i.i.i = phi double [ %19, %2 ], [ %101, %21 ]
  %.pn3.i.i.i = phi double [ %17, %2 ], [ %99, %21 ]
  %.pn1.in.i.i.i = phi i32 [ %20, %2 ], [ %88, %21 ]
  %102 = fmul double %.pn3.i.i.i, %.pn3.i.i.i
  %103 = fmul double %102, 5.000000e-01
  %104 = fsub double 1.000000e+00, %103
  %105 = fsub double 1.000000e+00, %104
  %106 = fsub double %105, %103
  %107 = fmul double %102, %102
  %108 = tail call double @llvm.fma.f64(double %102, double 0xBDA907DB46CC5E42, double 0x3E21EEB69037AB78)
  %109 = tail call double @llvm.fma.f64(double %102, double %108, double 0xBE927E4FA17F65F6)
  %110 = tail call double @llvm.fma.f64(double %102, double %109, double 0x3EFA01A019F4EC90)
  %111 = tail call double @llvm.fma.f64(double %102, double %110, double 0xBF56C16C16C16967)
  %112 = tail call double @llvm.fma.f64(double %102, double %111, double 0x3FA5555555555555)
  %113 = fneg double %.pn5.i.i.i
  %114 = tail call double @llvm.fma.f64(double %.pn3.i.i.i, double %113, double %106)
  %115 = tail call double @llvm.fma.f64(double %107, double %112, double %114)
  %116 = fadd double %104, %115
  %117 = tail call double @llvm.fma.f64(double %102, double 0x3DE5E0B2F9A43BB8, double 0xBE5AE600B42FDFA7)
  %118 = tail call double @llvm.fma.f64(double %102, double %117, double 0x3EC71DE3796CDE01)
  %119 = tail call double @llvm.fma.f64(double %102, double %118, double 0xBF2A01A019E83E5C)
  %120 = tail call double @llvm.fma.f64(double %102, double %119, double 0x3F81111111110BB3)
  %121 = fneg double %102
  %122 = fmul double %.pn3.i.i.i, %121
  %123 = fmul double %.pn5.i.i.i, 5.000000e-01
  %124 = tail call double @llvm.fma.f64(double %122, double %120, double %123)
  %125 = tail call double @llvm.fma.f64(double %102, double %124, double %113)
  %126 = tail call double @llvm.fma.f64(double %122, double 0xBFC5555555555555, double %125)
  %127 = fsub double %.pn3.i.i.i, %126
  %.pn1.i.i.i = shl i32 %.pn1.in.i.i.i, 30
  %128 = and i32 %.pn1.i.i.i, -2147483648
  %129 = and i32 %.pn1.in.i.i.i, 1
  %130 = icmp eq i32 %129, 0
  %131 = select i1 %130, double %127, double %116
  %132 = bitcast double %131 to <2 x i32>
  %133 = bitcast double %_a.coerce1 to <2 x i32>
  %134 = extractelement <2 x i32> %133, i64 1
  %135 = extractelement <2 x i32> %132, i64 1
  %136 = xor i32 %.pn1.i.i.i, %134
  %137 = and i32 %136, -2147483648
  %138 = xor i32 %135, %137
  %139 = insertelement <2 x i32> %132, i32 %138, i64 1
  %140 = fneg double %127
  %141 = select i1 %130, double %116, double %140
  %142 = bitcast double %141 to <2 x i32>
  %143 = extractelement <2 x i32> %142, i64 1
  %144 = xor i32 %143, %128
  %145 = insertelement <2 x i32> %142, i32 %144, i64 1
  %146 = fcmp one double %0, 0x7FF0000000000000
  %147 = select i1 %146, <2 x i32> %139, <2 x i32> <i32 0, i32 2146959360>
  %148 = select i1 %146, <2 x i32> %145, <2 x i32> <i32 0, i32 2146959360>
  %149 = bitcast <2 x i32> %148 to double
  %150 = bitcast <2 x i32> %147 to double
  %151 = fcmp ogt double %_a.coerce0, 7.090000e+02
  %152 = select i1 %151, double 1.000000e+00, double 0.000000e+00
  %153 = fsub double %_a.coerce0, %152
  %154 = fmul double %153, 0x3FF71547652B82FE
  %155 = tail call double @llvm.rint.f64(double %154)
  %156 = fneg double %155
  %157 = tail call double @llvm.fma.f64(double %156, double 0x3FE62E42FEFA39EF, double %153)
  %158 = tail call double @llvm.fma.f64(double %156, double 0x3C7ABC9E3B39803F, double %157)
  %159 = tail call double @llvm.fma.f64(double %158, double 0x3E5ADE156A5DCB37, double 0x3E928AF3FCA7AB0C)
  %160 = tail call double @llvm.fma.f64(double %158, double %159, double 0x3EC71DEE623FDE64)
  %161 = tail call double @llvm.fma.f64(double %158, double %160, double 0x3EFA01997C89E6B0)
  %162 = tail call double @llvm.fma.f64(double %158, double %161, double 0x3F2A01A014761F6E)
  %163 = tail call double @llvm.fma.f64(double %158, double %162, double 0x3F56C16C1852B7B0)
  %164 = tail call double @llvm.fma.f64(double %158, double %163, double 0x3F81111111122322)
  %165 = tail call double @llvm.fma.f64(double %158, double %164, double 0x3FA55555555502A1)
  %166 = tail call double @llvm.fma.f64(double %158, double %165, double 0x3FC5555555555511)
  %167 = tail call double @llvm.fma.f64(double %158, double %166, double 0x3FE000000000000B)
  %168 = tail call double @llvm.fma.f64(double %158, double %167, double 1.000000e+00)
  %169 = tail call double @llvm.fma.f64(double %158, double %168, double 1.000000e+00)
  %170 = fptosi double %155 to i32
  %171 = tail call double @llvm.ldexp.f64.i32(double %169, i32 %170)
  %172 = fcmp ogt double %153, 1.024000e+03
  %173 = select i1 %172, double 0x7FF0000000000000, double %171
  %174 = fcmp olt double %153, -1.075000e+03
  %175 = select i1 %174, double 0.000000e+00, double %173
  %176 = fcmp uno double %_a.coerce0, 0.000000e+00
  %177 = fcmp oeq double %_a.coerce1, 0.000000e+00
  %178 = and i1 %176, %177
  %179 = fcmp oeq double %_a.coerce0, 0x7FF0000000000000
  %180 = fcmp oeq double %_a.coerce0, 0xFFF0000000000000
  %181 = select i1 %151, double 0x4005BF0A8B145769, double 1.000000e+00
  %182 = fmul double %181, %150
  %183 = fmul double %175, %182
  %184 = select i1 %146, double %183, double 0.000000e+00
  %185 = select i1 %180, double %184, double %183
  %186 = select i1 %146, double %185, double 0x7FF8000000000000
  %187 = select i1 %177, double %_a.coerce1, double %186
  %188 = select i1 %179, double %187, double %185
  %189 = select i1 %178, double %_a.coerce1, double %188
  %190 = fmul double %181, %149
  %191 = fmul double %175, %190
  %192 = select i1 %180, double 0.000000e+00, double %191
  %193 = select i1 %146, double %192, double 0x7FF0000000000000
  %194 = select i1 %179, double %193, double %192
  %.fca.0.insert = insertvalue { double, double } poison, double %194, 0
  %.fca.1.insert = insertvalue { double, double } %.fca.0.insert, double %189, 1
  ret { double, double } %.fca.1.insert
}

; Function Attrs: cold mustprogress noinline nounwind optsize
define weak hidden [2 x i32] @cexpf([2 x i32] noundef %_a.coerce) local_unnamed_addr #11 {
entry:
  %_a.coerce.fca.1.extract = extractvalue [2 x i32] %_a.coerce, 1
  %0 = bitcast i32 %_a.coerce.fca.1.extract to float
  %1 = tail call float @llvm.fabs.f32(float %0) #14
  %2 = fcmp olt float %1, 1.310720e+05
  br i1 %2, label %3, label %10

3:                                                ; preds = %entry
  %4 = fmul float %1, 0x3FE45F3060000000
  %5 = tail call float @llvm.rint.f32(float %4) #14
  %6 = tail call float @llvm.fma.f32(float %5, float 0xBFF921FB40000000, float %1) #14
  %7 = tail call float @llvm.fma.f32(float %5, float 0xBE74442D00000000, float %6) #14
  %8 = tail call float @llvm.fma.f32(float %5, float 0xBCF8469880000000, float %7) #14
  %9 = fptosi float %5 to i32
  %.pre.i.i = bitcast float %1 to i32
  br label %__ocml_cexp_f32.exit

10:                                               ; preds = %entry
  %11 = bitcast float %1 to i32
  %12 = lshr i32 %11, 23
  %13 = add nsw i32 %12, -120
  %14 = icmp ugt i32 %13, 63
  %15 = select i1 %14, i32 -64, i32 0
  %16 = add nsw i32 %15, %13
  %17 = icmp ugt i32 %16, 31
  %18 = select i1 %17, i32 -32, i32 0
  %19 = add nsw i32 %18, %16
  %20 = icmp ugt i32 %19, 31
  %21 = select i1 %20, i32 -32, i32 0
  %22 = add nsw i32 %21, %19
  %23 = icmp eq i32 %22, 0
  %24 = and i32 %11, 8388607
  %25 = or disjoint i32 %24, 8388608
  %26 = zext nneg i32 %25 to i64
  %27 = mul nuw nsw i64 %26, 4266746795
  %28 = lshr i64 %27, 32
  %29 = mul nuw nsw i64 %26, 1011060801
  %30 = add nuw nsw i64 %28, %29
  %31 = lshr i64 %30, 32
  %32 = mul nuw nsw i64 %26, 3680671129
  %33 = add nuw nsw i64 %31, %32
  %34 = lshr i64 %33, 32
  %35 = mul nuw nsw i64 %26, 4113882560
  %36 = add nuw nsw i64 %34, %35
  %37 = trunc i64 %36 to i32
  %38 = lshr i64 %36, 32
  %39 = mul nuw nsw i64 %26, 4230436817
  %40 = add nuw nsw i64 %38, %39
  %41 = lshr i64 %40, 32
  %42 = mul nuw nsw i64 %26, 1313084713
  %43 = add nuw nsw i64 %41, %42
  %44 = trunc i64 %43 to i32
  %45 = select i1 %14, i32 %37, i32 %44
  %46 = trunc i64 %40 to i32
  %47 = lshr i64 %43, 32
  %48 = mul nuw nsw i64 %26, 2734261102
  %49 = add nuw nsw i64 %47, %48
  %50 = trunc i64 %49 to i32
  %51 = select i1 %14, i32 %46, i32 %50
  %52 = select i1 %17, i32 %45, i32 %51
  %53 = lshr i64 %49, 32
  %54 = trunc i64 %53 to i32
  %55 = select i1 %14, i32 %44, i32 %54
  %56 = select i1 %17, i32 %51, i32 %55
  %57 = select i1 %20, i32 %52, i32 %56
  %58 = trunc i64 %33 to i32
  %59 = select i1 %14, i32 %58, i32 %46
  %60 = select i1 %17, i32 %59, i32 %45
  %61 = select i1 %20, i32 %60, i32 %52
  %62 = sub nsw i32 32, %22
  %63 = tail call i32 @llvm.fshr.i32(i32 %57, i32 %61, i32 %62) #14
  %64 = select i1 %23, i32 %57, i32 %63
  %65 = trunc i64 %30 to i32
  %66 = select i1 %14, i32 %65, i32 %37
  %67 = select i1 %17, i32 %66, i32 %59
  %68 = select i1 %20, i32 %67, i32 %60
  %69 = tail call i32 @llvm.fshr.i32(i32 %61, i32 %68, i32 %62) #14
  %70 = select i1 %23, i32 %61, i32 %69
  %71 = tail call i32 @llvm.fshl.i32(i32 %64, i32 %70, i32 2) #14
  %72 = lshr i32 %64, 29
  %73 = and i32 %72, 1
  %74 = sub nsw i32 0, %73
  %75 = xor i32 %71, %74
  %76 = trunc i64 %27 to i32
  %77 = select i1 %14, i32 %76, i32 %58
  %78 = select i1 %17, i32 %77, i32 %66
  %79 = select i1 %20, i32 %78, i32 %67
  %80 = tail call i32 @llvm.fshr.i32(i32 %68, i32 %79, i32 %62) #14
  %81 = select i1 %23, i32 %68, i32 %80
  %82 = tail call i32 @llvm.fshl.i32(i32 %70, i32 %81, i32 2) #14
  %83 = xor i32 %82, %74
  %84 = tail call i32 @llvm.ctlz.i32(i32 %75, i1 false) #14, !range !20
  %85 = sub nsw i32 31, %84
  %86 = tail call i32 @llvm.fshr.i32(i32 %75, i32 %83, i32 %85) #14
  %87 = tail call i32 @llvm.fshl.i32(i32 %81, i32 %79, i32 2) #14
  %88 = xor i32 %87, %74
  %89 = tail call i32 @llvm.fshr.i32(i32 %83, i32 %88, i32 %85) #14
  %90 = tail call i32 @llvm.fshl.i32(i32 %86, i32 %89, i32 23) #14
  %91 = tail call i32 @llvm.ctlz.i32(i32 %90, i1 false) #14, !range !20
  %92 = sub nsw i32 31, %91
  %93 = tail call i32 @llvm.fshr.i32(i32 %90, i32 %89, i32 %92) #14
  %94 = lshr i32 %93, 9
  %95 = add nuw nsw i32 %91, %84
  %96 = shl i32 %72, 31
  %97 = or disjoint i32 %96, 855638016
  %98 = shl nuw nsw i32 %95, 23
  %99 = sub nuw i32 %97, %98
  %100 = or disjoint i32 %99, %94
  %101 = bitcast i32 %100 to float
  %102 = lshr i32 %86, 9
  %103 = or disjoint i32 %96, 1056964608
  %104 = shl nuw nsw i32 %84, 23
  %105 = sub nuw nsw i32 %103, %104
  %106 = or disjoint i32 %102, %105
  %107 = bitcast i32 %106 to float
  %108 = fmul float %107, 0x3FF921FB40000000
  %109 = fneg float %108
  %110 = tail call float @llvm.fma.f32(float %107, float 0x3FF921FB40000000, float %109) #14
  %111 = tail call float @llvm.fma.f32(float %107, float 0x3E74442D00000000, float %110) #14
  %112 = tail call float @llvm.fma.f32(float %101, float 0x3FF921FB40000000, float %111) #14
  %113 = fadd float %108, %112
  %114 = lshr i32 %64, 30
  %115 = add nuw nsw i32 %73, %114
  br label %__ocml_cexp_f32.exit

__ocml_cexp_f32.exit:                             ; preds = %3, %10
  %.pre-phi.i.i = phi i32 [ %.pre.i.i, %3 ], [ %11, %10 ]
  %.pn3.in.i.i.i = phi float [ %8, %3 ], [ %113, %10 ]
  %.pn1.in.i.i.i = phi i32 [ %9, %3 ], [ %115, %10 ]
  %_a.coerce.fca.0.extract = extractvalue [2 x i32] %_a.coerce, 0
  %116 = bitcast i32 %_a.coerce.fca.0.extract to float
  %117 = fmul float %.pn3.in.i.i.i, %.pn3.in.i.i.i
  %118 = tail call noundef float @llvm.fmuladd.f32(float %117, float 0xBF29833040000000, float 0x3F81103880000000)
  %119 = tail call noundef float @llvm.fmuladd.f32(float %117, float %118, float 0xBFC55553A0000000)
  %120 = fmul float %117, %119
  %121 = tail call noundef float @llvm.fmuladd.f32(float %.pn3.in.i.i.i, float %120, float %.pn3.in.i.i.i)
  %122 = tail call noundef float @llvm.fmuladd.f32(float %117, float 0x3EFAEA6680000000, float 0xBF56C9E760000000)
  %123 = tail call noundef float @llvm.fmuladd.f32(float %117, float %122, float 0x3FA5557EE0000000)
  %124 = tail call noundef float @llvm.fmuladd.f32(float %117, float %123, float 0xBFE0000080000000)
  %125 = tail call noundef float @llvm.fmuladd.f32(float %117, float %124, float 1.000000e+00)
  %.pn1.i.i.i = shl i32 %.pn1.in.i.i.i, 30
  %126 = and i32 %.pn1.i.i.i, -2147483648
  %127 = and i32 %.pn1.in.i.i.i, 1
  %128 = icmp eq i32 %127, 0
  %129 = select i1 %128, float %121, float %125
  %130 = bitcast float %129 to i32
  %131 = xor i32 %.pre-phi.i.i, %130
  %132 = xor i32 %131, %_a.coerce.fca.1.extract
  %133 = xor i32 %132, %126
  %134 = bitcast i32 %133 to float
  %135 = fneg float %121
  %136 = select i1 %128, float %125, float %135
  %137 = bitcast float %136 to i32
  %138 = xor i32 %126, %137
  %139 = bitcast i32 %138 to float
  %140 = fcmp one float %1, 0x7FF0000000000000
  %141 = select i1 %140, float %139, float 0x7FF8000000000000
  %142 = select i1 %140, float %134, float 0x7FF8000000000000
  %143 = fcmp ogt float %116, 8.800000e+01
  %144 = select i1 %143, float 1.000000e+00, float 0.000000e+00
  %145 = fsub float %116, %144
  %146 = tail call noundef float @llvm.exp.f32(float %145)
  %147 = fcmp uno float %116, 0.000000e+00
  %148 = fcmp oeq float %0, 0.000000e+00
  %149 = and i1 %147, %148
  %150 = fcmp oeq float %116, 0x7FF0000000000000
  %151 = fcmp oeq float %116, 0xFFF0000000000000
  %152 = select i1 %143, float 0x4005BF0A80000000, float 1.000000e+00
  %153 = fmul float %152, %142
  %154 = fmul float %146, %153
  %155 = select i1 %140, float %154, float 0.000000e+00
  %156 = select i1 %151, float %155, float %154
  %157 = select i1 %140, float %156, float 0x7FF8000000000000
  %158 = select i1 %148, float %0, float %157
  %159 = select i1 %150, float %158, float %156
  %160 = select i1 %149, float %0, float %159
  %161 = fmul float %152, %141
  %162 = fmul float %146, %161
  %163 = select i1 %151, float 0.000000e+00, float %162
  %164 = select i1 %140, float %163, float 0x7FF0000000000000
  %165 = select i1 %150, float %164, float %163
  %166 = bitcast float %165 to i32
  %.fca.0.insert = insertvalue [2 x i32] poison, i32 %166, 0
  %167 = bitcast float %160 to i32
  %.fca.1.insert = insertvalue [2 x i32] %.fca.0.insert, i32 %167, 1
  ret [2 x i32] %.fca.1.insert
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fabs.f64(double) #12

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.rint.f64(double) #12

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fma.f64(double, double, double) #12

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.ldexp.f64.i32(double, i32) #12

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.amdgcn.trig.preop.f64(double, i32) #12

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.floor.f64(double) #12

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.minnum.f64(double, double) #12

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fabs.f32(float) #12

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.exp.f32(float) #12

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.fshr.i32(i32, i32, i32) #12

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.fshl.i32(i32, i32, i32) #12

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.ctlz.i32(i32, i1 immarg) #12

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.fma.f32(float, float, float) #12

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.rint.f32(float) #12

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.copysign.f64(double, double) #12

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.copysign.f32(float, float) #12

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.maxnum.f64(double, double) #12

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.maxnum.f32(float, float) #12

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare { double, i32 } @llvm.frexp.f64.i32(double) #12

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare float @llvm.ldexp.f32.i32(float, i32) #12

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare i32 @llvm.smin.i32(i32, i32) #12

attributes #0 = { alwaysinline norecurse nounwind "amdgpu-flat-work-group-size"="1,256" "kernel" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx908" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" "uniform-work-group-size"="true" }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #2 = { nounwind }
attributes #3 = { alwaysinline norecurse nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx908" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" }
attributes #4 = { norecurse nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx908" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" }
attributes #5 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }
attributes #6 = { convergent norecurse nounwind "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx908" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" }
attributes #7 = { convergent nounwind }
attributes #8 = { alwaysinline }
attributes #9 = { mustprogress nofree norecurse nosync nounwind willreturn memory(readwrite, inaccessiblemem: none) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx908" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" }
attributes #10 = { mustprogress nofree norecurse nosync nounwind willreturn memory(read, argmem: readwrite, inaccessiblemem: none) "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx908" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" }
attributes #11 = { cold mustprogress noinline nounwind optsize "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx908" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+mai-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" }
attributes #12 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #13 = { nounwind memory(readwrite) }
attributes #14 = { nosync }

!omp_offload.info = !{!0}
!nvvm.annotations = !{!1}
!llvm.module.flags = !{!2, !3, !4, !5, !6}
!opencl.ocl.version = !{!7}
!llvm.ident = !{!8}

!0 = !{i32 0, i32 64768, i32 69609006, !"main", i32 15, i32 0, i32 0}
!1 = !{ptr @__omp_offloading_fd00_426262e_main_l15, !"kernel", i32 1}
!2 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 7, !"openmp", i32 51}
!5 = !{i32 7, !"openmp-device", i32 51}
!6 = !{i32 8, !"PIC Level", i32 2}
!7 = !{i32 2, i32 0}
!8 = !{!"AOMP_STANDALONE_19.0-0 clang version 19.0.0_AOMP_STANDALONE_19.0-0 (ssh://nicebert@gerrit-git.amd.com:29418/lightning/ec/llvm-project 4ee36e59440d581921c7e1d782a08208cf536cf0)"}
!9 = !{!10}
!10 = distinct !{!10, !11, !"__omp_offloading_fd00_426262e_main_l15_omp_outlined: %.global_tid."}
!11 = distinct !{!11, !"__omp_offloading_fd00_426262e_main_l15_omp_outlined"}
!12 = !{!13, !13, i64 0}
!13 = !{!"int", !14, i64 0}
!14 = !{!"omnipotent char", !15, i64 0}
!15 = !{!"Simple C++ TBAA"}
!16 = !{!17, !17, i64 0}
!17 = !{!"any pointer", !14, i64 0}
!18 = !{i64 0, i64 16, !19}
!19 = !{!14, !14, i64 0}
!20 = !{i32 0, i32 33}

