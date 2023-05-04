; RUN: opt -S -passes=openmp-opt < %s | FileCheck %s
;
; Verify we change it to SPMD mode but also avoid propagating the old mode (=generic) into the __kmpc_target_init function.
;
; CHECK-NOT: store i32 0, ptr addrspace(3) @IsSPMDMode
; CHECK: call i32 @__kmpc_target_init(ptr addrspacecast (ptr addrspace(1) @1 to ptr), i8 2, i1 false)
; CHECK-NOT: store i32 0, ptr addrspace(3) @IsSPMDMode
; CHECK: store i32 1, ptr addrspace(3) @IsSPMDMode
; CHECK-NOT: store i32 0, ptr addrspace(3) @IsSPMDMode
;
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8"
target triple = "amdgcn-amd-amdhsa"

%struct.ident_t = type { i32, i32, i32, i32, ptr }
%struct.DeviceEnvironmentTy = type { i32, i32, i32, i32 }
%"struct.(anonymous namespace)::SharedMemorySmartStackTy" = type { [512 x i8], [1024 x i8] }
%"struct.(anonymous namespace)::TeamStateTy" = type { %"struct.(anonymous namespace)::ICVStateTy", i32, ptr }
%"struct.(anonymous namespace)::ICVStateTy" = type { i32, i32, i32, i32, i32, i32 }

@__omp_rtl_assume_teams_oversubscription = weak_odr hidden local_unnamed_addr addrspace(1) constant i32 0
@__omp_rtl_assume_threads_oversubscription = weak_odr hidden local_unnamed_addr addrspace(1) constant i32 0
@0 = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00", align 1
@1 = private unnamed_addr addrspace(1) constant %struct.ident_t { i32 0, i32 2, i32 0, i32 22, ptr @0 }, align 8
@__omp_offloading_20_11e3950_main_l12_exec_mode = weak addrspace(1) constant i8 1
@__omp_rtl_debug_kind = weak_odr hidden local_unnamed_addr addrspace(1) constant i32 0
@__omp_rtl_assume_no_thread_state = weak_odr hidden local_unnamed_addr addrspace(1) constant i32 0
@omptarget_device_environment = weak protected addrspace(4) global %struct.DeviceEnvironmentTy undef, align 4
@IsSPMDMode = weak hidden addrspace(3) global i32 undef, align 4
@.str.12 = private unnamed_addr addrspace(4) constant [47 x i8] c"ValueRAII initialization with wrong old value!\00", align 1
@_ZN12_GLOBAL__N_122SharedMemorySmartStackE = internal addrspace(3) global %"struct.(anonymous namespace)::SharedMemorySmartStackTy" undef, align 16
@_ZN12_GLOBAL__N_19TeamStateE = internal unnamed_addr addrspace(3) global %"struct.(anonymous namespace)::TeamStateTy" undef, align 8
@_ZN12_GLOBAL__N_112ThreadStatesE = internal unnamed_addr addrspace(3) global [1024 x ptr] undef, align 16
@.str.12.47 = private unnamed_addr addrspace(4) constant [49 x i8] c"Thread state modified while explicitly disabled!\00", align 1
@_ZL29SharedMemVariableSharingSpace = internal unnamed_addr addrspace(3) global [64 x ptr] undef, align 16
@G = global i32 undef
@llvm.used = appending addrspace(1) global [2 x ptr] [ptr addrspacecast (ptr addrspace(3) @IsSPMDMode to ptr), ptr addrspacecast (ptr addrspace(4) @omptarget_device_environment to ptr)], section "llvm.metadata"
@llvm.compiler.used = appending addrspace(1) global [1 x ptr] [ptr addrspacecast (ptr addrspace(1) @__omp_offloading_20_11e3950_main_l12_exec_mode to ptr)], section "llvm.metadata"

; Function Attrs: alwaysinline convergent norecurse nounwind
define weak_odr amdgpu_kernel void @__omp_offloading_20_11e3950_main_l12(i64 noundef %nxyz, i64 noundef %ng, ptr noundef nonnull align 8 dereferenceable(8) %aa) local_unnamed_addr #0 {
entry:
  %ng1 = alloca i32, align 4
  %captured_vars_addrs = alloca [2 x ptr], align 8, addrspace(5)
  %0 = call i32 @__kmpc_target_init(ptr addrspacecast (ptr addrspace(1) @1 to ptr), i8 1, i1 true)
  %exec_user_code = icmp eq i32 %0, -1
  br i1 %exec_user_code, label %user_code.entry, label %common.ret

user_code.entry:                                  ; preds = %entry
  %captured_vars_addrs.ascast = addrspacecast ptr addrspace(5) %captured_vars_addrs to ptr
  store ptr %ng1, ptr addrspace(5) %captured_vars_addrs, align 8, !tbaa !7
  call void @__kmpc_parallel_51(ptr addrspacecast (ptr addrspace(1) @1 to ptr), i32 0, i32 1, i32 -1, i32 -1, ptr nonnull @__omp_outlined__, ptr nonnull @__omp_outlined___wrapper, ptr nonnull %captured_vars_addrs.ascast, i64 2)
  call void @__kmpc_target_deinit(ptr addrspacecast (ptr addrspace(1) @1 to ptr), i8 1)
  br label %common.ret

common.ret:                                       ; preds = %user_code.entry, %entry
  ret void
}

; Function Attrs: argmemonly nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.start.p5(i64 immarg, ptr addrspace(5) nocapture) #1

; Function Attrs: alwaysinline mustprogress nofree norecurse nosync nounwind readnone willreturn
define internal void @__omp_outlined__(ptr noalias nocapture %.global_tid., ptr noalias nocapture %.bound_tid., ptr nocapture nonnull align 4 %ng, ptr nocapture nonnull align 8 %aa) #2 {
entry:
  %isspmd = load i32, ptr addrspace(3) @IsSPMDMode
  store i32 %isspmd, ptr @G
  ret void
}

; Function Attrs: argmemonly nocallback nofree nosync nounwind willreturn
declare void @llvm.lifetime.end.p5(i64 immarg, ptr addrspace(5) nocapture) #1

; Function Attrs: mustprogress nofree norecurse nosync nounwind readnone willreturn
define internal void @__omp_outlined___wrapper(i16 zeroext %0, i32 noundef %1) #3 {
entry:
  %isspmd = load i32, ptr addrspace(3) @IsSPMDMode
  store i32 %isspmd, ptr @G
  ret void
}

; Function Attrs: nounwind readnone speculatable willreturn
declare i32 @llvm.amdgcn.workitem.id.x() #4

; Function Attrs: nounwind readnone speculatable willreturn
declare i32 @llvm.amdgcn.workgroup.id.x() #4

; Function Attrs: nounwind readnone speculatable willreturn
declare align 4 ptr addrspace(4) @llvm.amdgcn.dispatch.ptr() #4

; Function Attrs: nocallback nofree nosync nounwind readnone speculatable willreturn
declare i32 @llvm.umin.i32(i32, i32) #5

; Function Attrs: inaccessiblememonly nocallback nofree nosync nounwind willreturn
declare void @llvm.assume(i1 noundef) #6

; Function Attrs: convergent nounwind willreturn
declare void @llvm.amdgcn.s.barrier() #7

; Function Attrs: convergent mustprogress noinline nounwind willreturn
define internal fastcc void @_ZN4ompx11synchronize14threadsAlignedEv() unnamed_addr #8 {
entry:
  call void @llvm.amdgcn.s.barrier() #13
  ret void
}

; Function Attrs: convergent nounwind
define internal i32 @__kmpc_target_init(ptr nocapture noundef readnone %Ident, i8 noundef signext %Mode, i1 noundef zeroext %UseGenericStateMachine) local_unnamed_addr #9 {
entry:
  %0 = and i32 undef, undef
  %1 = and i8 %Mode, 2
  %tobool.not = icmp eq i8 %1, 0
  br i1 %tobool.not, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  %2 = call i32 @llvm.amdgcn.workitem.id.x() #13, !range !11
  %3 = call i32 @llvm.amdgcn.workgroup.id.x() #13
  %4 = call align 4 dereferenceable(64) ptr addrspace(4) @llvm.amdgcn.dispatch.ptr() #13
  %5 = getelementptr i8, ptr addrspace(4) %4, i64 12
  %6 = load i32, ptr addrspace(4) %5, align 4, !invariant.load !12
  %7 = getelementptr i8, ptr addrspace(4) %4, i64 4
  %8 = load i16, ptr addrspace(4) %7, align 4, !range !13, !invariant.load !12
  %conv.i.i7.i.i.i = zext i16 %8 to i32
  %mul.i.i8.i.i.i = mul i32 %3, %conv.i.i7.i.i.i
  %sub.i.i9.i.i.i = sub i32 %6, %mul.i.i8.i.i.i
  %9 = call i32 @llvm.umin.i32(i32 %sub.i.i9.i.i.i, i32 %conv.i.i7.i.i.i) #13
  %cmp4.i.i.i = icmp ult i32 %2, %9
  call void @llvm.assume(i1 %cmp4.i.i.i) #13
  %cmp.i.i8 = icmp eq i32 %2, 0
  br i1 %cmp.i.i8, label %if.then.i, label %_ZN4ompx5state4initEb.exit.critedge

if.then.i:                                        ; preds = %if.then
  store i32 1, ptr addrspace(3) @IsSPMDMode, align 4, !tbaa !14
  store i8 0, ptr addrspace(3) getelementptr inbounds (%"struct.(anonymous namespace)::SharedMemorySmartStackTy", ptr addrspace(3) @_ZN12_GLOBAL__N_122SharedMemorySmartStackE, i32 0, i32 1, i32 0), align 16, !tbaa !18
  store i32 %9, ptr addrspace(3) @_ZN12_GLOBAL__N_19TeamStateE, align 8, !tbaa !19
  store i32 0, ptr addrspace(3) getelementptr inbounds (%"struct.(anonymous namespace)::TeamStateTy", ptr addrspace(3) @_ZN12_GLOBAL__N_19TeamStateE, i32 0, i32 0, i32 1), align 4, !tbaa !23
  store i32 0, ptr addrspace(3) getelementptr inbounds (%"struct.(anonymous namespace)::TeamStateTy", ptr addrspace(3) @_ZN12_GLOBAL__N_19TeamStateE, i32 0, i32 0, i32 2), align 8, !tbaa !24
  store i32 1, ptr addrspace(3) getelementptr inbounds (%"struct.(anonymous namespace)::TeamStateTy", ptr addrspace(3) @_ZN12_GLOBAL__N_19TeamStateE, i32 0, i32 0, i32 3), align 4, !tbaa !25
  store i32 1, ptr addrspace(3) getelementptr inbounds (%"struct.(anonymous namespace)::TeamStateTy", ptr addrspace(3) @_ZN12_GLOBAL__N_19TeamStateE, i32 0, i32 0, i32 4), align 8, !tbaa !26
  store i32 1, ptr addrspace(3) getelementptr inbounds (%"struct.(anonymous namespace)::TeamStateTy", ptr addrspace(3) @_ZN12_GLOBAL__N_19TeamStateE, i32 0, i32 0, i32 5), align 4, !tbaa !27
  store i32 1, ptr addrspace(3) getelementptr inbounds (%"struct.(anonymous namespace)::TeamStateTy", ptr addrspace(3) @_ZN12_GLOBAL__N_19TeamStateE, i32 0, i32 1), align 8, !tbaa !28
  store ptr null, ptr addrspace(3) getelementptr inbounds (%"struct.(anonymous namespace)::TeamStateTy", ptr addrspace(3) @_ZN12_GLOBAL__N_19TeamStateE, i32 0, i32 2), align 8, !tbaa !29
  br label %_ZN4ompx5state4initEb.exit

_ZN4ompx5state4initEb.exit.critedge:              ; preds = %if.then
  %arrayidx.i.i.c = getelementptr inbounds [1024 x i8], ptr addrspace(3) getelementptr inbounds (%"struct.(anonymous namespace)::SharedMemorySmartStackTy", ptr addrspace(3) @_ZN12_GLOBAL__N_122SharedMemorySmartStackE, i32 0, i32 1, i32 0), i32 0, i32 %2
  store i8 0, ptr addrspace(3) %arrayidx.i.i.c, align 1, !tbaa !18
  br label %_ZN4ompx5state4initEb.exit

_ZN4ompx5state4initEb.exit:                       ; preds = %_ZN4ompx5state4initEb.exit.critedge, %if.then.i
  %arrayidx.i = getelementptr inbounds [1024 x ptr], ptr addrspace(3) @_ZN12_GLOBAL__N_112ThreadStatesE, i32 0, i32 %2
  store ptr null, ptr addrspace(3) %arrayidx.i, align 8, !tbaa !30
  call fastcc void @_ZN4ompx11synchronize14threadsAlignedEv() #14
  br label %if.end

if.else:                                          ; preds = %entry
  %10 = call i32 @llvm.amdgcn.workgroup.id.x() #13
  %11 = call align 4 dereferenceable(64) ptr addrspace(4) @llvm.amdgcn.dispatch.ptr() #13
  %12 = getelementptr i8, ptr addrspace(4) %11, i64 12
  %13 = load i32, ptr addrspace(4) %12, align 4, !invariant.load !12
  %14 = getelementptr i8, ptr addrspace(4) %11, i64 4
  %15 = load i16, ptr addrspace(4) %14, align 4, !range !13, !invariant.load !12
  %conv.i.i.i.i.i.i = zext i16 %15 to i32
  %mul.i.i.i.i.i.i = mul i32 %10, %conv.i.i.i.i.i.i
  %sub.i.i.i.i.i.i = sub i32 %13, %mul.i.i.i.i.i.i
  %16 = call i32 @llvm.umin.i32(i32 %sub.i.i.i.i.i.i, i32 %conv.i.i.i.i.i.i) #13
  %17 = call i32 @llvm.amdgcn.workitem.id.x() #13
  %cmp.i.i.i.i26 = icmp ult i32 %17, %16
  call void @llvm.assume(i1 %cmp.i.i.i.i26) #13
  %sub.i.i.i27 = add nsw i32 %16, -1
  %and.i.i.i28 = and i32 %sub.i.i.i27, -64
  %cmp.i2.i.i29 = icmp eq i32 %17, %and.i.i.i28
  br i1 %cmp.i2.i.i29, label %if.then.i30, label %_ZN4ompx5state4initEb.exit55.critedge

if.then.i30:                                      ; preds = %if.else
  store i32 0, ptr addrspace(3) @IsSPMDMode, align 4, !tbaa !14
  %arrayidx.i.i46 = getelementptr inbounds [1024 x i8], ptr addrspace(3) getelementptr inbounds (%"struct.(anonymous namespace)::SharedMemorySmartStackTy", ptr addrspace(3) @_ZN12_GLOBAL__N_122SharedMemorySmartStackE, i32 0, i32 1, i32 0), i32 0, i32 %17
  store i8 0, ptr addrspace(3) %arrayidx.i.i46, align 1, !tbaa !18
  %sub.i.i = add nsw i32 %16, -64
  store i32 %sub.i.i, ptr addrspace(3) @_ZN12_GLOBAL__N_19TeamStateE, align 8, !tbaa !19
  store i32 0, ptr addrspace(3) getelementptr inbounds (%"struct.(anonymous namespace)::TeamStateTy", ptr addrspace(3) @_ZN12_GLOBAL__N_19TeamStateE, i32 0, i32 0, i32 1), align 4, !tbaa !23
  store i32 0, ptr addrspace(3) getelementptr inbounds (%"struct.(anonymous namespace)::TeamStateTy", ptr addrspace(3) @_ZN12_GLOBAL__N_19TeamStateE, i32 0, i32 0, i32 2), align 8, !tbaa !24
  store i32 1, ptr addrspace(3) getelementptr inbounds (%"struct.(anonymous namespace)::TeamStateTy", ptr addrspace(3) @_ZN12_GLOBAL__N_19TeamStateE, i32 0, i32 0, i32 3), align 4, !tbaa !25
  store i32 1, ptr addrspace(3) getelementptr inbounds (%"struct.(anonymous namespace)::TeamStateTy", ptr addrspace(3) @_ZN12_GLOBAL__N_19TeamStateE, i32 0, i32 0, i32 4), align 8, !tbaa !26
  store i32 1, ptr addrspace(3) getelementptr inbounds (%"struct.(anonymous namespace)::TeamStateTy", ptr addrspace(3) @_ZN12_GLOBAL__N_19TeamStateE, i32 0, i32 0, i32 5), align 4, !tbaa !27
  store i32 1, ptr addrspace(3) getelementptr inbounds (%"struct.(anonymous namespace)::TeamStateTy", ptr addrspace(3) @_ZN12_GLOBAL__N_19TeamStateE, i32 0, i32 1), align 8, !tbaa !28
  store ptr null, ptr addrspace(3) getelementptr inbounds (%"struct.(anonymous namespace)::TeamStateTy", ptr addrspace(3) @_ZN12_GLOBAL__N_19TeamStateE, i32 0, i32 2), align 8, !tbaa !29
  br label %_ZN4ompx5state4initEb.exit55

_ZN4ompx5state4initEb.exit55.critedge:            ; preds = %if.else
  %arrayidx.i.i46.c = getelementptr inbounds [1024 x i8], ptr addrspace(3) getelementptr inbounds (%"struct.(anonymous namespace)::SharedMemorySmartStackTy", ptr addrspace(3) @_ZN12_GLOBAL__N_122SharedMemorySmartStackE, i32 0, i32 1, i32 0), i32 0, i32 %17
  store i8 0, ptr addrspace(3) %arrayidx.i.i46.c, align 1, !tbaa !18
  br label %_ZN4ompx5state4initEb.exit55

_ZN4ompx5state4initEb.exit55:                     ; preds = %_ZN4ompx5state4initEb.exit55.critedge, %if.then.i30
  %arrayidx.i53 = getelementptr inbounds [1024 x ptr], ptr addrspace(3) @_ZN12_GLOBAL__N_112ThreadStatesE, i32 0, i32 %17
  store ptr null, ptr addrspace(3) %arrayidx.i53, align 8, !tbaa !30
  br label %if.end

if.end:                                           ; preds = %_ZN4ompx5state4initEb.exit55, %_ZN4ompx5state4initEb.exit
  %18 = call i32 @llvm.amdgcn.workgroup.id.x() #13
  %19 = call align 4 dereferenceable(64) ptr addrspace(4) @llvm.amdgcn.dispatch.ptr() #13
  %20 = getelementptr i8, ptr addrspace(4) %19, i64 12
  %21 = load i32, ptr addrspace(4) %20, align 4, !invariant.load !12
  %22 = getelementptr i8, ptr addrspace(4) %19, i64 4
  %23 = load i16, ptr addrspace(4) %22, align 4, !range !13, !invariant.load !12
  %conv.i.i.i.i.i73 = zext i16 %23 to i32
  %mul.i.i.i.i.i74 = mul i32 %18, %conv.i.i.i.i.i73
  %sub.i.i.i.i.i75 = sub i32 %21, %mul.i.i.i.i.i74
  %24 = call i32 @llvm.umin.i32(i32 %sub.i.i.i.i.i75, i32 %conv.i.i.i.i.i73) #13
  %25 = call i32 @llvm.amdgcn.workitem.id.x() #13
  %cmp.i.i.i79 = icmp ult i32 %25, %24
  call void @llvm.assume(i1 %cmp.i.i.i79) #13
  br i1 %tobool.not, label %_ZN4ompx7mapping23isInitialThreadInLevel0Eb.exit, label %_ZN4ompx7mapping12getBlockSizeEb.exit.i64

_ZN4ompx7mapping12getBlockSizeEb.exit.i64:        ; preds = %if.end
  %26 = load i32, ptr addrspace(3) @_ZN12_GLOBAL__N_19TeamStateE, align 8
  %cmp.i.i.i63 = icmp eq i32 %24, %26
  call void @llvm.assume(i1 %cmp.i.i.i63) #13
  %27 = load i32, ptr addrspace(3) getelementptr inbounds (%"struct.(anonymous namespace)::TeamStateTy", ptr addrspace(3) @_ZN12_GLOBAL__N_19TeamStateE, i32 0, i32 0, i32 1), align 4
  %cmp9.i.i.i = icmp eq i32 %27, 0
  call void @llvm.assume(i1 %cmp9.i.i.i) #13
  %28 = load i32, ptr addrspace(3) getelementptr inbounds (%"struct.(anonymous namespace)::TeamStateTy", ptr addrspace(3) @_ZN12_GLOBAL__N_19TeamStateE, i32 0, i32 0, i32 2), align 8
  %cmp19.i.i.i = icmp eq i32 %28, 0
  call void @llvm.assume(i1 %cmp19.i.i.i) #13
  %29 = load i32, ptr addrspace(3) getelementptr inbounds (%"struct.(anonymous namespace)::TeamStateTy", ptr addrspace(3) @_ZN12_GLOBAL__N_19TeamStateE, i32 0, i32 0, i32 3), align 4
  %cmp29.i.i.i = icmp eq i32 %29, 1
  call void @llvm.assume(i1 %cmp29.i.i.i) #13
  %30 = load i32, ptr addrspace(3) getelementptr inbounds (%"struct.(anonymous namespace)::TeamStateTy", ptr addrspace(3) @_ZN12_GLOBAL__N_19TeamStateE, i32 0, i32 0, i32 4), align 8
  %cmp39.i.i.i = icmp eq i32 %30, 1
  call void @llvm.assume(i1 %cmp39.i.i.i) #13
  %31 = load i32, ptr addrspace(3) getelementptr inbounds (%"struct.(anonymous namespace)::TeamStateTy", ptr addrspace(3) @_ZN12_GLOBAL__N_19TeamStateE, i32 0, i32 0, i32 5), align 4
  %cmp49.i.i.i = icmp eq i32 %31, 1
  call void @llvm.assume(i1 %cmp49.i.i.i) #13
  %32 = load i32, ptr addrspace(3) getelementptr inbounds (%"struct.(anonymous namespace)::TeamStateTy", ptr addrspace(3) @_ZN12_GLOBAL__N_19TeamStateE, i32 0, i32 1), align 8
  %cmp.i.i67 = icmp eq i32 %32, 1
  call void @llvm.assume(i1 %cmp.i.i67) #13
  %33 = load i32, ptr addrspace(3) @IsSPMDMode, align 4, !tbaa !14
  %tobool.i59.i = icmp ne i32 %33, 0
  call void @llvm.assume(i1 %tobool.i59.i) #13
  br label %_ZN14DebugEntryRAIID2Ev.exit250

_ZN4ompx7mapping23isInitialThreadInLevel0Eb.exit: ; preds = %if.end
  %sub.i.i83 = add nsw i32 %24, -1
  %and.i.i84 = and i32 %sub.i.i83, -64
  %cmp.i2.i = icmp eq i32 %25, %and.i.i84
  br i1 %cmp.i2.i, label %_ZN14DebugEntryRAIID2Ev.exit250, label %if.end10

if.end10:                                         ; preds = %_ZN4ompx7mapping23isInitialThreadInLevel0Eb.exit
  %sub.i = add nsw i32 %24, -64
  %cmp = icmp ult i32 %25, %sub.i
  %or.cond251 = select i1 %UseGenericStateMachine, i1 %cmp, i1 false
  br i1 %or.cond251, label %do.body.i, label %_ZN14DebugEntryRAIID2Ev.exit250

do.body.i:                                        ; preds = %if.end10
  call void @llvm.amdgcn.s.barrier() #13
  br label %_ZN14DebugEntryRAIID2Ev.exit250

_ZN14DebugEntryRAIID2Ev.exit250:                  ; preds = %do.body.i, %if.end10, %_ZN4ompx7mapping23isInitialThreadInLevel0Eb.exit, %_ZN4ompx7mapping12getBlockSizeEb.exit.i64
  %retval.0 = phi i32 [ -1, %_ZN4ompx7mapping12getBlockSizeEb.exit.i64 ], [ -1, %_ZN4ompx7mapping23isInitialThreadInLevel0Eb.exit ], [ %25, %do.body.i ], [ %25, %if.end10 ]
  ret i32 %retval.0
}

; Function Attrs: nounwind
define internal void @__kmpc_target_deinit(ptr nocapture noundef readnone %Ident, i8 noundef signext %Mode) local_unnamed_addr #10 {
  ret void
}

; Function Attrs: convergent nounwind
declare void @__kmpc_parallel_51(ptr nocapture noundef readnone %ident, i32 noundef %0, i32 noundef %if_expr, i32 noundef %num_threads, i32 noundef %proc_bind, ptr noundef %fn, ptr noundef %wrapper_fn, ptr noundef %args, i64 noundef %nargs)

; Function Attrs: argmemonly nofree nounwind willreturn
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #12

attributes #0 = { alwaysinline convergent norecurse nounwind "frame-pointer"="none" "kernel" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx908" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+flat-address-space,+gfx8-insts,+gfx9-insts,+mai-insts,+s-memrealtime,+s-memtime-inst" }
attributes #1 = { argmemonly nocallback nofree nosync nounwind willreturn }
attributes #2 = { alwaysinline mustprogress nofree norecurse nosync nounwind readnone willreturn "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx908" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+flat-address-space,+gfx8-insts,+gfx9-insts,+mai-insts,+s-memrealtime,+s-memtime-inst" }
attributes #3 = { mustprogress nofree norecurse nosync nounwind readnone willreturn "frame-pointer"="none" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx908" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+flat-address-space,+gfx8-insts,+gfx9-insts,+mai-insts,+s-memrealtime,+s-memtime-inst" }
attributes #4 = { nounwind readnone speculatable willreturn }
attributes #5 = { nocallback nofree nosync nounwind readnone speculatable willreturn }
attributes #6 = { inaccessiblememonly nocallback nofree nosync nounwind willreturn }
attributes #7 = { convergent nounwind willreturn }
attributes #8 = { convergent mustprogress noinline nounwind willreturn "frame-pointer"="none" "llvm.assume"="ompx_aligned_barrier,ompx_no_call_asm,ompx_no_call_asm" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx908" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+flat-address-space,+gfx8-insts,+gfx9-insts,+mai-insts,+s-memrealtime,+s-memtime-inst" }
attributes #9 = { convergent nounwind "frame-pointer"="none" "llvm.assume"="ompx_no_call_asm,ompx_no_call_asm" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx908" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+flat-address-space,+gfx8-insts,+gfx9-insts,+mai-insts,+s-memrealtime,+s-memtime-inst" }
attributes #10 = { nounwind "frame-pointer"="none" "llvm.assume"="ompx_no_call_asm,ompx_no_call_asm" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx908" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+flat-address-space,+gfx8-insts,+gfx9-insts,+mai-insts,+s-memrealtime,+s-memtime-inst" }
attributes #11 = { convergent nounwind "frame-pointer"="none" "llvm.assume"="ompx_no_call_asm" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx908" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot2-insts,+dot3-insts,+dot4-insts,+dot5-insts,+dot6-insts,+dot7-insts,+dpp,+flat-address-space,+gfx8-insts,+gfx9-insts,+mai-insts,+s-memrealtime,+s-memtime-inst" }
attributes #12 = { argmemonly nofree nounwind willreturn }
attributes #13 = { nounwind }
attributes #14 = { convergent nounwind "llvm.assume"="ompx_aligned_barrier,ompx_no_call_asm" }
attributes #15 = { convergent nounwind }

!omp_offload.info = !{!0}
!nvvm.annotations = !{!1}
!llvm.module.flags = !{!2, !3, !4, !5}
!llvm.ident = !{!6}

!0 = !{i32 0, i32 32, i32 18757968, !"main", i32 12, i32 0}
!1 = !{ptr @__omp_offloading_20_11e3950_main_l12, !"kernel", i32 1}
!2 = !{i32 1, !"wchar_size", i32 4}
!3 = !{i32 7, !"openmp", i32 50}
!4 = !{i32 7, !"openmp-device", i32 50}
!5 = !{i32 7, !"PIC Level", i32 2}
!6 = !{!"clang version 15.0.0"}
!7 = !{!8, !8, i64 0}
!8 = !{!"any pointer", !9, i64 0}
!9 = !{!"omnipotent char", !10, i64 0}
!10 = !{!"Simple C/C++ TBAA"}
!11 = !{i32 0, i32 1024}
!12 = !{}
!13 = !{i16 1, i16 1025}
!14 = !{!15, !15, i64 0}
!15 = !{!"int", !16, i64 0}
!16 = !{!"omnipotent char", !17, i64 0}
!17 = !{!"Simple C++ TBAA"}
!18 = !{!16, !16, i64 0}
!19 = !{!20, !15, i64 0}
!20 = !{!"_ZTSN12_GLOBAL__N_111TeamStateTyE", !21, i64 0, !15, i64 24, !22, i64 32}
!21 = !{!"_ZTSN12_GLOBAL__N_110ICVStateTyE", !15, i64 0, !15, i64 4, !15, i64 8, !15, i64 12, !15, i64 16, !15, i64 20}
!22 = !{!"any pointer", !16, i64 0}
!23 = !{!20, !15, i64 4}
!24 = !{!20, !15, i64 8}
!25 = !{!20, !15, i64 12}
!26 = !{!20, !15, i64 16}
!27 = !{!20, !15, i64 20}
!28 = !{!20, !15, i64 24}
!29 = !{!20, !22, i64 32}
!30 = !{!22, !22, i64 0}
!31 = !{!"branch_weights", i32 2000, i32 1}
