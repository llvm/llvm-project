; See ./README.md for how to maintain the LLVM IR in this test.

; REQUIRES: nvptx-registered-target

; RUN: opt -pass-remarks=kernel-info -passes=kernel-info \
; RUN:     -disable-output %s 2>&1 | \
; RUN:   FileCheck -match-full-lines %s

;  CHECK-NOT: remark:
;      CHECK: remark: test.c:0:0: in artificial function '[[OFF_FUNC:__omp_offloading_[a-f0-9_]*_h_l12]]_debug__', artificial alloca 'dyn_ptr' with static size of 8 bytes
; CHECK-NEXT: remark: test.c:14:9: in artificial function '[[OFF_FUNC]]_debug__', alloca 'i' with static size of 4 bytes
; CHECK-NEXT: remark: test.c:15:9: in artificial function '[[OFF_FUNC]]_debug__', alloca 'a' with static size of 8 bytes
; CHECK-NEXT: remark: <unknown>:0:0: in artificial function '[[OFF_FUNC]]_debug__', 'store' instruction accesses memory in flat address space
; CHECK-NEXT: remark: test.c:13:3: in artificial function '[[OFF_FUNC]]_debug__', direct call to defined function, callee is '@__kmpc_target_init'
; CHECK-NEXT: remark: test.c:16:5: in artificial function '[[OFF_FUNC]]_debug__', direct call, callee is '@f'
; CHECK-NEXT: remark: test.c:17:5: in artificial function '[[OFF_FUNC]]_debug__', direct call to defined function, callee is 'g'
; CHECK-NEXT: remark: test.c:18:3: in artificial function '[[OFF_FUNC]]_debug__', direct call to defined function, callee is '@__kmpc_target_deinit'
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', ExternalNotKernel = 0
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', omp_target_thread_limit = 128
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', maxntidx = 128
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', Allocas = 3
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', AllocasStaticSizeSum = 20
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', AllocasDyn = 0
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', DirectCalls = 4
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', IndirectCalls = 0
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', DirectCallsToDefinedFunctions = 3
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', InlineAssemblyCalls = 0
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', Invokes = 0
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', FlatAddrspaceAccesses = 1

; CHECK-NEXT: remark: test.c:0:0: in artificial function '[[OFF_FUNC]]', artificial alloca 'dyn_ptr' with static size of 8 bytes
; CHECK-NEXT: remark: <unknown>:0:0: in artificial function '[[OFF_FUNC]]', 'store' instruction accesses memory in flat address space
; CHECK-NEXT: remark: test.c:12:1: in artificial function '[[OFF_FUNC]]', 'load' instruction ('%[[#]]') accesses memory in flat address space
; CHECK-NEXT: remark: test.c:12:1: in artificial function '[[OFF_FUNC]]', direct call to defined function, callee is artificial '[[OFF_FUNC]]_debug__'
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', ExternalNotKernel = 0
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', Allocas = 1
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', AllocasStaticSizeSum = 8
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', AllocasDyn = 0
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', DirectCalls = 1
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', IndirectCalls = 0
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', DirectCallsToDefinedFunctions = 1
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', InlineAssemblyCalls = 0
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', Invokes = 0
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', FlatAddrspaceAccesses = 2

; CHECK-NEXT: remark: test.c:4:7: in function 'g', alloca 'i' with static size of 4 bytes
; CHECK-NEXT: remark: test.c:5:7: in function 'g', alloca 'a' with static size of 8 bytes
; CHECK-NEXT: remark: test.c:6:3: in function 'g', direct call, callee is '@f'
; CHECK-NEXT: remark: test.c:7:3: in function 'g', direct call to defined function, callee is 'g'
; CHECK-NEXT: remark: test.c:3:0: in function 'g', ExternalNotKernel = 1
; CHECK-NEXT: remark: test.c:3:0: in function 'g', Allocas = 2
; CHECK-NEXT: remark: test.c:3:0: in function 'g', AllocasStaticSizeSum = 12
; CHECK-NEXT: remark: test.c:3:0: in function 'g', AllocasDyn = 0
; CHECK-NEXT: remark: test.c:3:0: in function 'g', DirectCalls = 2
; CHECK-NEXT: remark: test.c:3:0: in function 'g', IndirectCalls = 0
; CHECK-NEXT: remark: test.c:3:0: in function 'g', DirectCallsToDefinedFunctions = 1
; CHECK-NEXT: remark: test.c:3:0: in function 'g', InlineAssemblyCalls = 0
; CHECK-NEXT: remark: test.c:3:0: in function 'g', Invokes = 0
; CHECK-NEXT: remark: test.c:3:0: in function 'g', FlatAddrspaceAccesses = 0
;  CHECK-NOT: remark: {{.*: in function 'g',.*}}

; A lot of internal functions (e.g., __kmpc_target_init) come next, but we don't
; want to maintain a list of their allocas, calls, etc. in this test.


; ModuleID = 'test-openmp-nvptx64-nvidia-cuda.bc'
source_filename = "test.c"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

%struct.ident_t = type { i32, i32, i32, i32, ptr }
%struct.DynamicEnvironmentTy = type { i16 }
%struct.KernelEnvironmentTy = type { %struct.ConfigurationEnvironmentTy, ptr, ptr }
%struct.ConfigurationEnvironmentTy = type { i8, i8, i8, i32, i32, i32, i32, i32, i32 }
%struct.DeviceMemoryPoolTy = type { ptr, i64 }
%struct.DeviceMemoryPoolTrackingTy = type { i64, i64, i64, i64 }
%struct.DeviceEnvironmentTy = type { i32, i32, i32, i32, i64, i64, i64, i64 }
%"struct.(anonymous namespace)::SharedMemorySmartStackTy" = type { [512 x i8], [1024 x i8] }
%"struct.ompx::state::TeamStateTy" = type { %"struct.ompx::state::ICVStateTy", i32, i32, ptr }
%"struct.ompx::state::ICVStateTy" = type { i32, i32, i32, i32, i32, i32, i32 }
%printf_args = type { ptr, i32, ptr, ptr, ptr }
%printf_args.7 = type { ptr, i32, ptr, ptr }

@__omp_rtl_assume_teams_oversubscription = weak_odr hidden constant i32 0
@__omp_rtl_assume_threads_oversubscription = weak_odr hidden constant i32 0
@0 = private unnamed_addr constant [59 x i8] c";test.c;__omp_offloading_10305_5c00dd_h_l12_debug__;13;3;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 58, ptr @0 }, align 8
@__omp_offloading_10305_5c00dd_h_l12_dynamic_environment = weak_odr protected global %struct.DynamicEnvironmentTy zeroinitializer
@__omp_offloading_10305_5c00dd_h_l12_kernel_environment = weak_odr protected constant %struct.KernelEnvironmentTy { %struct.ConfigurationEnvironmentTy { i8 1, i8 1, i8 1, i32 1, i32 128, i32 -1, i32 -1, i32 0, i32 0 }, ptr @1, ptr @__omp_offloading_10305_5c00dd_h_l12_dynamic_environment }
@llvm.used = appending global [3 x ptr] [ptr addrspacecast (ptr addrspace(4) @__omp_rtl_device_environment to ptr), ptr @__omp_rtl_device_memory_pool, ptr @__omp_rtl_device_memory_pool_tracker], section "llvm.metadata"
@__omp_rtl_device_memory_pool = weak protected global %struct.DeviceMemoryPoolTy zeroinitializer, align 8
@__omp_rtl_device_memory_pool_tracker = weak protected global %struct.DeviceMemoryPoolTrackingTy zeroinitializer, align 8
@__omp_rtl_debug_kind = weak_odr hidden constant i32 0
@__omp_rtl_assume_no_thread_state = weak_odr hidden constant i32 0
@__omp_rtl_assume_no_nested_parallelism = weak_odr hidden constant i32 0
@__omp_rtl_device_environment = weak protected addrspace(4) global %struct.DeviceEnvironmentTy undef, align 8
@.str = private unnamed_addr constant [40 x i8] c"%s:%u: %s: Assertion %s (`%s`) failed.\0A\00", align 1
@.str1 = private unnamed_addr constant [35 x i8] c"%s:%u: %s: Assertion `%s` failed.\0A\00", align 1
@.str15 = private unnamed_addr constant [43 x i8] c"/tmp/llvm/offload/DeviceRTL/src/Kernel.cpp\00", align 1
@__PRETTY_FUNCTION__._ZL19genericStateMachineP7IdentTy = private unnamed_addr constant [36 x i8] c"void genericStateMachine(IdentTy *)\00", align 1
@.str2 = private unnamed_addr constant [18 x i8] c"WorkFn == nullptr\00", align 1
@__PRETTY_FUNCTION__.__kmpc_target_deinit = private unnamed_addr constant [28 x i8] c"void __kmpc_target_deinit()\00", align 1
@IsSPMDMode = internal local_unnamed_addr addrspace(3) global i32 undef, align 4
@.str1127 = private unnamed_addr constant [48 x i8] c"/tmp/llvm/offload/DeviceRTL/src/Parallelism.cpp\00", align 1
@.str13 = private unnamed_addr constant [23 x i8] c"!mapping::isSPMDMode()\00", align 1
@__PRETTY_FUNCTION__.__kmpc_kernel_end_parallel = private unnamed_addr constant [34 x i8] c"void __kmpc_kernel_end_parallel()\00", align 1
@_ZL20KernelEnvironmentPtr = internal unnamed_addr addrspace(3) global ptr undef, align 8
@_ZL26KernelLaunchEnvironmentPtr = internal unnamed_addr addrspace(3) global ptr undef, align 8
@_ZN12_GLOBAL__N_122SharedMemorySmartStackE = internal addrspace(3) global %"struct.(anonymous namespace)::SharedMemorySmartStackTy" undef, align 16
@.str544 = private unnamed_addr constant [42 x i8] c"/tmp/llvm/offload/DeviceRTL/src/State.cpp\00", align 1
@.str847 = private unnamed_addr constant [33 x i8] c"NThreadsVar == Other.NThreadsVar\00", align 1
@__PRETTY_FUNCTION__._ZNK4ompx5state10ICVStateTy11assertEqualERKS1_ = private unnamed_addr constant [68 x i8] c"void ompx::state::ICVStateTy::assertEqual(const ICVStateTy &) const\00", align 1
@.str948 = private unnamed_addr constant [27 x i8] c"LevelVar == Other.LevelVar\00", align 1
@.str1049 = private unnamed_addr constant [39 x i8] c"ActiveLevelVar == Other.ActiveLevelVar\00", align 1
@.str1150 = private unnamed_addr constant [47 x i8] c"MaxActiveLevelsVar == Other.MaxActiveLevelsVar\00", align 1
@.str1251 = private unnamed_addr constant [33 x i8] c"RunSchedVar == Other.RunSchedVar\00", align 1
@.str1352 = private unnamed_addr constant [43 x i8] c"RunSchedChunkVar == Other.RunSchedChunkVar\00", align 1
@.str14 = private unnamed_addr constant [43 x i8] c"ParallelTeamSize == Other.ParallelTeamSize\00", align 1
@__PRETTY_FUNCTION__._ZNK4ompx5state11TeamStateTy11assertEqualERS1_ = private unnamed_addr constant [64 x i8] c"void ompx::state::TeamStateTy::assertEqual(TeamStateTy &) const\00", align 1
@.str1553 = private unnamed_addr constant [39 x i8] c"HasThreadState == Other.HasThreadState\00", align 1
@.str24 = private unnamed_addr constant [32 x i8] c"mapping::isSPMDMode() == IsSPMD\00", align 1
@__PRETTY_FUNCTION__._ZN4ompx5state18assumeInitialStateEb = private unnamed_addr constant [43 x i8] c"void ompx::state::assumeInitialState(bool)\00", align 1
@_ZN4ompx5state9TeamStateE = internal local_unnamed_addr addrspace(3) global %"struct.ompx::state::TeamStateTy" undef, align 8
@_ZN4ompx5state12ThreadStatesE = internal addrspace(3) global ptr undef, align 8

; Function Attrs: convergent noinline norecurse nounwind optnone
define internal void @__omp_offloading_10305_5c00dd_h_l12_debug__(ptr noalias noundef %dyn_ptr) #0 !dbg !17 {
entry:
  %dyn_ptr.addr = alloca ptr, align 8
  %i = alloca i32, align 4
  %a = alloca [2 x i32], align 4
  store ptr %dyn_ptr, ptr %dyn_ptr.addr, align 8
  tail call void @llvm.dbg.declare(metadata ptr %dyn_ptr.addr, metadata !24, metadata !DIExpression()), !dbg !25
  %0 = call i32 @__kmpc_target_init(ptr @__omp_offloading_10305_5c00dd_h_l12_kernel_environment, ptr %dyn_ptr), !dbg !26
  %exec_user_code = icmp eq i32 %0, -1, !dbg !26
  br i1 %exec_user_code, label %user_code.entry, label %worker.exit, !dbg !26

user_code.entry:                                  ; preds = %entry
  tail call void @llvm.dbg.declare(metadata ptr %i, metadata !27, metadata !DIExpression()), !dbg !30
  tail call void @llvm.dbg.declare(metadata ptr %a, metadata !31, metadata !DIExpression()), !dbg !35
  call void @f() #16, !dbg !36
  call void @g() #16, !dbg !37
  call void @__kmpc_target_deinit(), !dbg !38
  ret void, !dbg !39

worker.exit:                                      ; preds = %entry
  ret void, !dbg !26
}

; Function Attrs: convergent
declare void @f(...) #1

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define weak_odr protected void @__omp_offloading_10305_5c00dd_h_l12(ptr noalias noundef %dyn_ptr) #2 !dbg !40 {
entry:
  %dyn_ptr.addr = alloca ptr, align 8
  store ptr %dyn_ptr, ptr %dyn_ptr.addr, align 8
  tail call void @llvm.dbg.declare(metadata ptr %dyn_ptr.addr, metadata !41, metadata !DIExpression()), !dbg !42
  %0 = load ptr, ptr %dyn_ptr.addr, align 8, !dbg !43
  call void @__omp_offloading_10305_5c00dd_h_l12_debug__(ptr %0) #17, !dbg !43
  ret void, !dbg !43
}

; Function Attrs: convergent noinline nounwind optnone
define hidden void @g() #3 !dbg !44 {
entry:
  %i = alloca i32, align 4
  %a = alloca [2 x i32], align 4
  tail call void @llvm.dbg.declare(metadata ptr %i, metadata !47, metadata !DIExpression()), !dbg !48
  tail call void @llvm.dbg.declare(metadata ptr %a, metadata !49, metadata !DIExpression()), !dbg !50
  call void @f() #16, !dbg !51
  call void @g() #16, !dbg !52
  ret void, !dbg !53
}

; Function Attrs: convergent mustprogress nounwind
define internal noundef i32 @__kmpc_target_init(ptr nofree noundef nonnull align 8 dereferenceable(48) %KernelEnvironment, ptr nofree noundef nonnull align 8 dereferenceable(16) %KernelLaunchEnvironment) #4 {
entry:
  %WorkFn.i = alloca ptr, align 8
  %ExecMode = getelementptr inbounds i8, ptr %KernelEnvironment, i64 2
  %0 = load i8, ptr %ExecMode, align 2, !tbaa !54
  %1 = and i8 %0, 2
  %tobool.not = icmp eq i8 %1, 0
  %2 = load i8, ptr %KernelEnvironment, align 8, !tbaa !60
  %tobool3.not = icmp ne i8 %2, 0
  br i1 %tobool.not, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  %3 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #18
  %cmp.i.i.i = icmp eq i32 %3, 0
  br i1 %cmp.i.i.i, label %if.then.i, label %_ZN4ompx5state4initEbR19KernelEnvironmentTyR25KernelLaunchEnvironmentTy.exit.critedge

if.then.i:                                        ; preds = %if.then
  store i32 1, ptr addrspace(3) @IsSPMDMode, align 4, !tbaa !61
  %idxprom.i.i = zext nneg i32 %3 to i64
  %arrayidx.i.i = getelementptr inbounds [1024 x i8], ptr getelementptr inbounds (i8, ptr addrspacecast (ptr addrspace(3) @_ZN12_GLOBAL__N_122SharedMemorySmartStackE to ptr), i64 512), i64 0, i64 %idxprom.i.i
  %4 = addrspacecast ptr %arrayidx.i.i to ptr addrspace(3)
  store i8 0, ptr addrspace(3) %4, align 1, !tbaa !62
  store i32 0, ptr addrspace(3) @_ZN4ompx5state9TeamStateE, align 8, !tbaa !63
  store i32 0, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 4) to ptr addrspace(3)), align 4, !tbaa !67
  store i32 0, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 8) to ptr addrspace(3)), align 8, !tbaa !68
  store i32 0, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 12) to ptr addrspace(3)), align 4, !tbaa !69
  store i32 1, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 16) to ptr addrspace(3)), align 8, !tbaa !70
  store i32 1, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 20) to ptr addrspace(3)), align 4, !tbaa !71
  store i32 1, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 24) to ptr addrspace(3)), align 8, !tbaa !72
  store i32 1, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 28) to ptr addrspace(3)), align 4, !tbaa !73
  store i32 0, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 32) to ptr addrspace(3)), align 8, !tbaa !74
  store ptr null, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 40) to ptr addrspace(3)), align 8, !tbaa !75
  store ptr null, ptr addrspace(3) @_ZN4ompx5state12ThreadStatesE, align 8, !tbaa !76
  store ptr %KernelEnvironment, ptr addrspace(3) @_ZL20KernelEnvironmentPtr, align 8, !tbaa !76
  store ptr %KernelLaunchEnvironment, ptr addrspace(3) @_ZL26KernelLaunchEnvironmentPtr, align 8, !tbaa !76
  br label %_ZN4ompx5state4initEbR19KernelEnvironmentTyR25KernelLaunchEnvironmentTy.exit

_ZN4ompx5state4initEbR19KernelEnvironmentTyR25KernelLaunchEnvironmentTy.exit.critedge: ; preds = %if.then
  %idxprom.i.i.c = zext i32 %3 to i64
  %arrayidx.i.i.c = getelementptr inbounds [1024 x i8], ptr getelementptr inbounds (i8, ptr addrspacecast (ptr addrspace(3) @_ZN12_GLOBAL__N_122SharedMemorySmartStackE to ptr), i64 512), i64 0, i64 %idxprom.i.i.c
  %5 = addrspacecast ptr %arrayidx.i.i.c to ptr addrspace(3)
  store i8 0, ptr addrspace(3) %5, align 1, !tbaa !62
  br label %_ZN4ompx5state4initEbR19KernelEnvironmentTyR25KernelLaunchEnvironmentTy.exit

_ZN4ompx5state4initEbR19KernelEnvironmentTyR25KernelLaunchEnvironmentTy.exit: ; preds = %_ZN4ompx5state4initEbR19KernelEnvironmentTyR25KernelLaunchEnvironmentTy.exit.critedge, %if.then.i
  tail call void @_ZN4ompx11synchronize14threadsAlignedENS_6atomic10OrderingTyE(i32 poison) #19
  br label %if.end

if.else:                                          ; preds = %entry
  %6 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #18
  %sub.i.i.i7 = add i32 %6, -1
  %and.i.i.i8 = and i32 %sub.i.i.i7, -32
  %7 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #18
  %cmp.i.i.i9 = icmp eq i32 %7, %and.i.i.i8
  br i1 %cmp.i.i.i9, label %if.then.i11, label %if.end.critedge

if.then.i11:                                      ; preds = %if.else
  store i32 0, ptr addrspace(3) @IsSPMDMode, align 4, !tbaa !61
  %idxprom.i.i13 = zext i32 %7 to i64
  %arrayidx.i.i14 = getelementptr inbounds [1024 x i8], ptr getelementptr inbounds (i8, ptr addrspacecast (ptr addrspace(3) @_ZN12_GLOBAL__N_122SharedMemorySmartStackE to ptr), i64 512), i64 0, i64 %idxprom.i.i13
  %8 = addrspacecast ptr %arrayidx.i.i14 to ptr addrspace(3)
  store i8 0, ptr addrspace(3) %8, align 1, !tbaa !62
  store i32 0, ptr addrspace(3) @_ZN4ompx5state9TeamStateE, align 8, !tbaa !63
  store i32 0, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 4) to ptr addrspace(3)), align 4, !tbaa !67
  store i32 0, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 8) to ptr addrspace(3)), align 8, !tbaa !68
  store i32 0, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 12) to ptr addrspace(3)), align 4, !tbaa !69
  store i32 1, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 16) to ptr addrspace(3)), align 8, !tbaa !70
  store i32 1, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 20) to ptr addrspace(3)), align 4, !tbaa !71
  store i32 1, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 24) to ptr addrspace(3)), align 8, !tbaa !72
  store i32 1, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 28) to ptr addrspace(3)), align 4, !tbaa !73
  store i32 0, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 32) to ptr addrspace(3)), align 8, !tbaa !74
  store ptr null, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 40) to ptr addrspace(3)), align 8, !tbaa !75
  store ptr null, ptr addrspace(3) @_ZN4ompx5state12ThreadStatesE, align 8, !tbaa !76
  store ptr %KernelEnvironment, ptr addrspace(3) @_ZL20KernelEnvironmentPtr, align 8, !tbaa !76
  store ptr %KernelLaunchEnvironment, ptr addrspace(3) @_ZL26KernelLaunchEnvironmentPtr, align 8, !tbaa !76
  br label %if.end

if.end.critedge:                                  ; preds = %if.else
  %idxprom.i.i13.c = zext i32 %7 to i64
  %arrayidx.i.i14.c = getelementptr inbounds [1024 x i8], ptr getelementptr inbounds (i8, ptr addrspacecast (ptr addrspace(3) @_ZN12_GLOBAL__N_122SharedMemorySmartStackE to ptr), i64 512), i64 0, i64 %idxprom.i.i13.c
  %9 = addrspacecast ptr %arrayidx.i.i14.c to ptr addrspace(3)
  store i8 0, ptr addrspace(3) %9, align 1, !tbaa !62
  br label %if.end

if.end:                                           ; preds = %if.end.critedge, %if.then.i11, %_ZN4ompx5state4initEbR19KernelEnvironmentTyR25KernelLaunchEnvironmentTy.exit
  br i1 %tobool.not, label %if.end9, label %if.then7

if.then7:                                         ; preds = %if.end
  %10 = load i32, ptr @__omp_rtl_debug_kind, align 4, !tbaa !61
  %11 = load i32, ptr addrspace(4) @__omp_rtl_device_environment, align 8, !tbaa !77
  %and.i.i.i21 = and i32 %10, 1
  %and.i.i = and i32 %and.i.i.i21, %11
  %tobool.i.i = icmp ne i32 %and.i.i, 0
  %.pre67.i.i.i = load i32, ptr addrspace(3) @_ZN4ompx5state9TeamStateE, align 8, !tbaa !80
  %cmp.i.i.i22 = icmp ne i32 %.pre67.i.i.i, 0
  %or.cond.not.i.i.i = select i1 %tobool.i.i, i1 %cmp.i.i.i22, i1 false
  br i1 %or.cond.not.i.i.i, label %if.then.i.i.i, label %if.else.i.i.i

if.then.i.i.i:                                    ; preds = %if.then7
  tail call fastcc void @__assert_fail_internal(ptr nofree noundef nonnull dereferenceable(33) @.str847, ptr noundef null, ptr nofree noundef nonnull dereferenceable(69) @.str544, i32 noundef 193, ptr nofree noundef nonnull dereferenceable(68) @__PRETTY_FUNCTION__._ZNK4ompx5state10ICVStateTy11assertEqualERKS1_) #20
  unreachable

if.else.i.i.i:                                    ; preds = %if.then7
  %cmp5.i.i.i = icmp eq i32 %.pre67.i.i.i, 0
  tail call void @llvm.assume(i1 noundef %cmp5.i.i.i) #21
  %12 = load i32, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 4) to ptr addrspace(3)), align 4, !tbaa !81
  br i1 %tobool.i.i, label %land.lhs.true7.i.i.i, label %if.else11.i.i.i

land.lhs.true7.i.i.i:                             ; preds = %if.else.i.i.i
  %cmp9.i.i.i = icmp eq i32 %12, 0
  br i1 %cmp9.i.i.i, label %if.else11.i.i.i, label %if.then10.i.i.i

if.then10.i.i.i:                                  ; preds = %land.lhs.true7.i.i.i
  tail call fastcc void @__assert_fail_internal(ptr nofree noundef nonnull dereferenceable(27) @.str948, ptr noundef null, ptr nofree noundef nonnull dereferenceable(69) @.str544, i32 noundef 194, ptr nofree noundef nonnull dereferenceable(68) @__PRETTY_FUNCTION__._ZNK4ompx5state10ICVStateTy11assertEqualERKS1_) #20
  unreachable

if.else11.i.i.i:                                  ; preds = %land.lhs.true7.i.i.i, %if.else.i.i.i
  %13 = phi i32 [ 0, %land.lhs.true7.i.i.i ], [ %12, %if.else.i.i.i ]
  %cmp14.i.i.i = icmp eq i32 %13, 0
  tail call void @llvm.assume(i1 noundef %cmp14.i.i.i) #21
  %14 = load i32, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 8) to ptr addrspace(3)), align 8, !tbaa !82
  br i1 %tobool.i.i, label %land.lhs.true17.i.i.i, label %if.else21.i.i.i

land.lhs.true17.i.i.i:                            ; preds = %if.else11.i.i.i
  %cmp19.i.i.i = icmp eq i32 %14, 0
  br i1 %cmp19.i.i.i, label %if.else21.i.i.i, label %if.then20.i.i.i

if.then20.i.i.i:                                  ; preds = %land.lhs.true17.i.i.i
  tail call fastcc void @__assert_fail_internal(ptr nofree noundef nonnull dereferenceable(39) @.str1049, ptr noundef null, ptr nofree noundef nonnull dereferenceable(69) @.str544, i32 noundef 195, ptr nofree noundef nonnull dereferenceable(68) @__PRETTY_FUNCTION__._ZNK4ompx5state10ICVStateTy11assertEqualERKS1_) #20
  unreachable

if.else21.i.i.i:                                  ; preds = %land.lhs.true17.i.i.i, %if.else11.i.i.i
  %15 = phi i32 [ 0, %land.lhs.true17.i.i.i ], [ %14, %if.else11.i.i.i ]
  %cmp24.i.i.i = icmp eq i32 %15, 0
  tail call void @llvm.assume(i1 noundef %cmp24.i.i.i) #21
  %16 = load i32, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 16) to ptr addrspace(3)), align 8, !tbaa !83
  br i1 %tobool.i.i, label %land.lhs.true27.i.i.i, label %if.else31.i.i.i

land.lhs.true27.i.i.i:                            ; preds = %if.else21.i.i.i
  %cmp29.i.i.i = icmp eq i32 %16, 1
  br i1 %cmp29.i.i.i, label %if.else31.i.i.i, label %if.then30.i.i.i

if.then30.i.i.i:                                  ; preds = %land.lhs.true27.i.i.i
  tail call fastcc void @__assert_fail_internal(ptr nofree noundef nonnull dereferenceable(47) @.str1150, ptr noundef null, ptr nofree noundef nonnull dereferenceable(69) @.str544, i32 noundef 196, ptr nofree noundef nonnull dereferenceable(68) @__PRETTY_FUNCTION__._ZNK4ompx5state10ICVStateTy11assertEqualERKS1_) #20
  unreachable

if.else31.i.i.i:                                  ; preds = %land.lhs.true27.i.i.i, %if.else21.i.i.i
  %17 = phi i32 [ 1, %land.lhs.true27.i.i.i ], [ %16, %if.else21.i.i.i ]
  %cmp34.i.i.i = icmp eq i32 %17, 1
  tail call void @llvm.assume(i1 noundef %cmp34.i.i.i) #21
  %18 = load i32, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 20) to ptr addrspace(3)), align 4, !tbaa !84
  br i1 %tobool.i.i, label %land.lhs.true37.i.i.i, label %if.else.critedge.i.critedge.critedge.critedge

land.lhs.true37.i.i.i:                            ; preds = %if.else31.i.i.i
  %cmp39.i.i.i = icmp eq i32 %18, 1
  br i1 %cmp39.i.i.i, label %if.else41.i.i.i, label %if.then40.i.i.i

if.then40.i.i.i:                                  ; preds = %land.lhs.true37.i.i.i
  tail call fastcc void @__assert_fail_internal(ptr nofree noundef nonnull dereferenceable(33) @.str1251, ptr noundef null, ptr nofree noundef nonnull dereferenceable(69) @.str544, i32 noundef 197, ptr nofree noundef nonnull dereferenceable(68) @__PRETTY_FUNCTION__._ZNK4ompx5state10ICVStateTy11assertEqualERKS1_) #20
  unreachable

if.else41.i.i.i:                                  ; preds = %land.lhs.true37.i.i.i
  %cmp44.i.i.i = icmp eq i32 1, 1
  tail call void @llvm.assume(i1 noundef %cmp44.i.i.i) #21
  br i1 %tobool.i.i, label %land.lhs.true47.i.i.i, label %if.else.critedge.i.critedge

land.lhs.true47.i.i.i:                            ; preds = %if.else41.i.i.i
  %19 = load i32, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 24) to ptr addrspace(3)), align 8, !tbaa !85
  %cmp49.i.i.i = icmp eq i32 %19, 1
  br i1 %cmp49.i.i.i, label %if.else51.i.i.i, label %if.then50.i.i.i

if.then50.i.i.i:                                  ; preds = %land.lhs.true47.i.i.i
  tail call fastcc void @__assert_fail_internal(ptr nofree noundef nonnull dereferenceable(43) @.str1352, ptr noundef null, ptr nofree noundef nonnull dereferenceable(69) @.str544, i32 noundef 198, ptr nofree noundef nonnull dereferenceable(68) @__PRETTY_FUNCTION__._ZNK4ompx5state10ICVStateTy11assertEqualERKS1_) #20
  unreachable

if.else51.i.i.i:                                  ; preds = %land.lhs.true47.i.i.i
  br i1 %tobool.i.i, label %land.lhs.true.i.i, label %if.else.critedge.i.critedge

land.lhs.true.i.i:                                ; preds = %if.else51.i.i.i
  %20 = load i32, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 28) to ptr addrspace(3)), align 4, !tbaa !73
  %cmp.i.i = icmp eq i32 %20, 1
  br i1 %cmp.i.i, label %land.lhs.true8.i.i, label %if.then.i.i

if.then.i.i:                                      ; preds = %land.lhs.true.i.i
  tail call fastcc void @__assert_fail_internal(ptr nofree noundef nonnull dereferenceable(43) @.str14, ptr noundef null, ptr nofree noundef nonnull dereferenceable(69) @.str544, i32 noundef 222, ptr nofree noundef nonnull dereferenceable(64) @__PRETTY_FUNCTION__._ZNK4ompx5state11TeamStateTy11assertEqualERS1_) #20
  unreachable

land.lhs.true8.i.i:                               ; preds = %land.lhs.true.i.i
  %21 = load i32, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 32) to ptr addrspace(3)), align 8, !tbaa !74
  %cmp10.i.i = icmp eq i32 %21, 0
  br i1 %cmp10.i.i, label %land.lhs.true.i24, label %if.then11.i.i

if.then11.i.i:                                    ; preds = %land.lhs.true8.i.i
  tail call fastcc void @__assert_fail_internal(ptr nofree noundef nonnull dereferenceable(39) @.str1553, ptr noundef null, ptr nofree noundef nonnull dereferenceable(69) @.str544, i32 noundef 223, ptr nofree noundef nonnull dereferenceable(64) @__PRETTY_FUNCTION__._ZNK4ompx5state11TeamStateTy11assertEqualERS1_) #20
  unreachable

land.lhs.true.i24:                                ; preds = %land.lhs.true8.i.i
  %22 = load i32, ptr addrspace(3) @IsSPMDMode, align 4, !tbaa !61
  %tobool.i25.i.not = icmp eq i32 %22, 0
  br i1 %tobool.i25.i.not, label %if.then.i25, label %_ZN4ompx5state18assumeInitialStateEb.exit

if.then.i25:                                      ; preds = %land.lhs.true.i24
  tail call fastcc void @__assert_fail_internal(ptr nofree noundef nonnull dereferenceable(32) @.str24, ptr noundef null, ptr nofree noundef nonnull dereferenceable(69) @.str544, i32 noundef 326, ptr nofree noundef nonnull dereferenceable(43) @__PRETTY_FUNCTION__._ZN4ompx5state18assumeInitialStateEb) #20
  unreachable

if.else.critedge.i.critedge.critedge.critedge:    ; preds = %if.else31.i.i.i
  %cmp44.i.i.i.c = icmp eq i32 %18, 1
  tail call void @llvm.assume(i1 noundef %cmp44.i.i.i.c) #21
  br label %if.else.critedge.i.critedge

if.else.critedge.i.critedge:                      ; preds = %if.else41.i.i.i, %if.else.critedge.i.critedge.critedge.critedge, %if.else51.i.i.i
  %.pre.i = load i32, ptr addrspace(3) @IsSPMDMode, align 4, !tbaa !61
  %23 = icmp ne i32 %.pre.i, 0
  br label %_ZN4ompx5state18assumeInitialStateEb.exit

_ZN4ompx5state18assumeInitialStateEb.exit:        ; preds = %land.lhs.true.i24, %if.else.critedge.i.critedge
  %cmp8.i = phi i1 [ %23, %if.else.critedge.i.critedge ], [ true, %land.lhs.true.i24 ]
  tail call void @llvm.assume(i1 noundef %cmp8.i) #21
  tail call void @_ZN4ompx11synchronize14threadsAlignedENS_6atomic10OrderingTyE(i32 poison) #19
  br label %cleanup

if.end9:                                          ; preds = %if.end
  %24 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #18
  %sub.i.i = add i32 %24, -1
  %and.i.i26 = and i32 %sub.i.i, -32
  %25 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #18
  %cmp.i.i27 = icmp eq i32 %25, %and.i.i26
  br i1 %cmp.i.i27, label %cleanup, label %if.end12

if.end12:                                         ; preds = %if.end9
  %sub.i = add i32 %24, -32
  %cmp = icmp ult i32 %25, %sub.i
  %or.cond33 = and i1 %tobool3.not, %cmp
  br i1 %or.cond33, label %do.body.i.preheader, label %cleanup

do.body.i.preheader:                              ; preds = %if.end12
  %26 = load i32, ptr @__omp_rtl_debug_kind, align 4
  %27 = load i32, ptr addrspace(4) @__omp_rtl_device_environment, align 8
  %and.i.i29 = and i32 %26, 1
  %and.i = and i32 %and.i.i29, %27
  %tobool.i = icmp ne i32 %and.i, 0
  br label %do.body.i

do.body.i:                                        ; preds = %do.body.i.preheader, %if.end9.i
  call void @llvm.lifetime.start.p0(i64 noundef 8, ptr noundef nonnull align 8 dereferenceable(8) %WorkFn.i) #22
  store ptr null, ptr %WorkFn.i, align 8, !tbaa !76
  tail call void @llvm.nvvm.barrier.sync(i32 noundef 8) #18
  %call1.i = call zeroext i1 @__kmpc_kernel_parallel(ptr noalias nocapture nofree noundef nonnull writeonly align 8 dereferenceable(8) %WorkFn.i) #22
  %28 = load ptr, ptr %WorkFn.i, align 8, !tbaa !76
  %tobool.not.not.i = icmp eq ptr %28, null
  br i1 %tobool.not.not.i, label %_ZL19genericStateMachineP7IdentTy.exit, label %if.end.i

if.end.i:                                         ; preds = %do.body.i
  br i1 %call1.i, label %if.then3.i, label %if.end9.i

if.then3.i:                                       ; preds = %if.end.i
  %29 = load i32, ptr addrspace(3) @IsSPMDMode, align 4
  %tobool.i30 = icmp ne i32 %29, 0
  %or.cond = select i1 %tobool.i, i1 %tobool.i30, i1 false
  br i1 %or.cond, label %if.then6.i, label %if.else.i

if.then6.i:                                       ; preds = %if.then3.i
  tail call fastcc void @__assert_fail_internal(ptr nofree noundef nonnull dereferenceable(23) @.str13, ptr noundef null, ptr nofree noundef nonnull dereferenceable(70) @.str15, i32 noundef 58, ptr nofree noundef nonnull dereferenceable(36) @__PRETTY_FUNCTION__._ZL19genericStateMachineP7IdentTy) #20
  unreachable

if.else.i:                                        ; preds = %if.then3.i
  %tobool.i31.not = icmp eq i32 %29, 0
  tail call void @llvm.assume(i1 noundef %tobool.i31.not) #21
  tail call void %28(i32 noundef 0, i32 noundef %25) #23
  tail call void @__kmpc_kernel_end_parallel() #24
  br label %if.end9.i

if.end9.i:                                        ; preds = %if.else.i, %if.end.i
  tail call void @llvm.nvvm.barrier.sync(i32 noundef 8) #18
  call void @llvm.lifetime.end.p0(i64 noundef 8, ptr noundef nonnull %WorkFn.i) #22
  br label %do.body.i, !llvm.loop !86

_ZL19genericStateMachineP7IdentTy.exit:           ; preds = %do.body.i
  call void @llvm.lifetime.end.p0(i64 noundef 8, ptr noundef nonnull %WorkFn.i) #22
  br label %cleanup

cleanup:                                          ; preds = %if.end12, %_ZL19genericStateMachineP7IdentTy.exit, %if.end9, %_ZN4ompx5state18assumeInitialStateEb.exit
  %retval.0 = phi i32 [ -1, %_ZN4ompx5state18assumeInitialStateEb.exit ], [ -1, %if.end9 ], [ %25, %_ZL19genericStateMachineP7IdentTy.exit ], [ %25, %if.end12 ]
  ret i32 %retval.0
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #5

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define internal void @_ZN4ompx11synchronize14threadsAlignedENS_6atomic10OrderingTyE(i32 %Ordering) local_unnamed_addr #6 {
entry:
  tail call void @llvm.nvvm.barrier0() #25
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #5

; Function Attrs: convergent mustprogress noreturn nounwind
define internal fastcc void @__assert_fail_internal(ptr nofree noundef nonnull dereferenceable(8) %expr, ptr noundef %msg, ptr nofree noundef nonnull dereferenceable(69) %file, i32 noundef %line, ptr nofree noundef nonnull dereferenceable(20) %function) unnamed_addr #7 {
entry:
  %tmp = alloca %printf_args, align 8
  %tmp1 = alloca %printf_args.7, align 8
  %tobool.not = icmp eq ptr %msg, null
  br i1 %tobool.not, label %if.else, label %if.then

if.then:                                          ; preds = %entry
  store ptr %file, ptr %tmp, align 8
  %0 = getelementptr inbounds i8, ptr %tmp, i64 8
  store i32 %line, ptr %0, align 8
  %1 = getelementptr inbounds i8, ptr %tmp, i64 16
  store ptr %function, ptr %1, align 8
  br label %if.end

if.else:                                          ; preds = %entry
  store ptr %file, ptr %tmp1, align 8
  %2 = getelementptr inbounds i8, ptr %tmp1, i64 8
  store i32 %line, ptr %2, align 8
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %.sink12 = phi i64 [ 16, %if.else ], [ 24, %if.then ]
  %tmp1.sink11 = phi ptr [ %tmp1, %if.else ], [ %tmp, %if.then ]
  %function.sink = phi ptr [ %function, %if.else ], [ %msg, %if.then ]
  %.sink9 = phi i64 [ 24, %if.else ], [ 32, %if.then ]
  %.str1.sink = phi ptr [ @.str1, %if.else ], [ @.str, %if.then ]
  %3 = getelementptr inbounds i8, ptr %tmp1.sink11, i64 %.sink12
  store ptr %function.sink, ptr %3, align 8
  %4 = getelementptr inbounds i8, ptr %tmp1.sink11, i64 %.sink9
  store ptr %expr, ptr %4, align 8
  %call.i.i = call noundef i32 @vprintf(ptr noundef nonnull %.str1.sink, ptr noundef nonnull %tmp1.sink11) #24
  call void @llvm.trap() #26
  unreachable
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #8

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #9

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.sync(i32) #10

; Function Attrs: convergent mustprogress nofree noinline norecurse nosync nounwind willreturn memory(read, argmem: write, inaccessiblemem: none)
define internal noundef zeroext i1 @__kmpc_kernel_parallel(ptr nocapture nofree noundef nonnull writeonly align 8 dereferenceable(8) %WorkFn) local_unnamed_addr #11 {
entry:
  %0 = load ptr, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 40) to ptr addrspace(3)), align 8, !tbaa !76
  store ptr %0, ptr %WorkFn, align 8, !tbaa !76
  %tobool.not = icmp eq ptr %0, null
  br i1 %tobool.not, label %return, label %if.end

if.end:                                           ; preds = %entry
  %1 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #27
  %2 = load i32, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 28) to ptr addrspace(3)), align 4, !tbaa !61
  %tobool.not.i = icmp eq i32 %2, 0
  %3 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #18
  %4 = load i32, ptr addrspace(3) @IsSPMDMode, align 4
  %tobool.i.not.i.i = icmp eq i32 %4, 0
  %mul.neg.i.i.i = select i1 %tobool.i.not.i.i, i32 -32, i32 0
  %sub.i.i.i = add i32 %mul.neg.i.i.i, %3
  %cond.i = select i1 %tobool.not.i, i32 %sub.i.i.i, i32 %2
  %cmp = icmp ult i32 %1, %cond.i
  br label %return

return:                                           ; preds = %if.end, %entry
  %retval.0 = phi i1 [ %cmp, %if.end ], [ false, %entry ]
  ret i1 %retval.0
}

; Function Attrs: convergent mustprogress noinline nounwind
define internal void @__kmpc_kernel_end_parallel() local_unnamed_addr #12 {
entry:
  %0 = load i32, ptr @__omp_rtl_debug_kind, align 4, !tbaa !61
  %1 = load i32, ptr addrspace(4) @__omp_rtl_device_environment, align 8, !tbaa !77
  %and.i.i = and i32 %0, 1
  %and.i = and i32 %and.i.i, %1
  %tobool.i = icmp ne i32 %and.i, 0
  %2 = load i32, ptr addrspace(3) @IsSPMDMode, align 4
  %tobool.i1 = icmp ne i32 %2, 0
  %or.cond = select i1 %tobool.i, i1 %tobool.i1, i1 false
  br i1 %or.cond, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  tail call fastcc void @__assert_fail_internal(ptr nofree noundef nonnull dereferenceable(23) @.str13, ptr noundef null, ptr nofree noundef nonnull dereferenceable(75) @.str1127, i32 noundef 297, ptr nofree noundef nonnull dereferenceable(34) @__PRETTY_FUNCTION__.__kmpc_kernel_end_parallel) #20
  unreachable

if.else:                                          ; preds = %entry
  %tobool.i2.not = icmp eq i32 %2, 0
  tail call void @llvm.assume(i1 noundef %tobool.i2.not) #21
  %3 = load i32, ptr @__omp_rtl_assume_no_thread_state, align 4, !tbaa !61
  %tobool.not.i.i = icmp eq i32 %3, 0
  %4 = load i32, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 32) to ptr addrspace(3)), align 8
  %tobool.not.i = icmp ne i32 %4, 0
  %or.cond.not.i = select i1 %tobool.not.i.i, i1 %tobool.not.i, i1 false
  br i1 %or.cond.not.i, label %lor.rhs.i, label %_ZN4ompx5state19resetStateForThreadEj.exit

lor.rhs.i:                                        ; preds = %if.else
  %5 = tail call noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #27
  %6 = load ptr, ptr addrspace(3) @_ZN4ompx5state12ThreadStatesE, align 8, !tbaa !76
  %idxprom.i = zext i32 %5 to i64
  %arrayidx.i = getelementptr inbounds ptr, ptr %6, i64 %idxprom.i
  %7 = load ptr, ptr %arrayidx.i, align 8, !tbaa !76
  %tobool1.not.i = icmp eq ptr %7, null
  br i1 %tobool1.not.i, label %_ZN4ompx5state19resetStateForThreadEj.exit, label %if.end4.i, !prof !88

if.end4.i:                                        ; preds = %lor.rhs.i
  %PreviousThreadState7.i = getelementptr inbounds i8, ptr %7, i64 32
  %8 = load ptr, ptr %PreviousThreadState7.i, align 8, !tbaa !89
  tail call void @free(ptr noundef nonnull dereferenceable(40) %7) #28
  %9 = load ptr, ptr addrspace(3) @_ZN4ompx5state12ThreadStatesE, align 8, !tbaa !76
  %arrayidx11.i = getelementptr inbounds ptr, ptr %9, i64 %idxprom.i
  store ptr %8, ptr %arrayidx11.i, align 8, !tbaa !76
  %.pre = load i32, ptr addrspace(3) @IsSPMDMode, align 4
  br label %_ZN4ompx5state19resetStateForThreadEj.exit

_ZN4ompx5state19resetStateForThreadEj.exit:       ; preds = %if.else, %lor.rhs.i, %if.end4.i
  %10 = phi i32 [ 0, %if.else ], [ 0, %lor.rhs.i ], [ %.pre, %if.end4.i ]
  %tobool.i6 = icmp ne i32 %10, 0
  %or.cond8 = select i1 %tobool.i, i1 %tobool.i6, i1 false
  br i1 %or.cond8, label %if.then7, label %if.else8

if.then7:                                         ; preds = %_ZN4ompx5state19resetStateForThreadEj.exit
  tail call fastcc void @__assert_fail_internal(ptr nofree noundef nonnull dereferenceable(23) @.str13, ptr noundef null, ptr nofree noundef nonnull dereferenceable(75) @.str1127, i32 noundef 300, ptr nofree noundef nonnull dereferenceable(34) @__PRETTY_FUNCTION__.__kmpc_kernel_end_parallel) #20
  unreachable

if.else8:                                         ; preds = %_ZN4ompx5state19resetStateForThreadEj.exit
  %tobool.i7.not = icmp eq i32 %10, 0
  tail call void @llvm.assume(i1 noundef %tobool.i7.not) #21
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #9

; Function Attrs: convergent mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite)
declare extern_weak void @free(ptr allocptr nocapture noundef) local_unnamed_addr #13

; Function Attrs: convergent
declare i32 @vprintf(ptr noundef, ptr noundef) local_unnamed_addr #14

; Function Attrs: cold noreturn nounwind memory(inaccessiblemem: write)
declare void @llvm.trap() #15

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier0() #10

; Function Attrs: convergent mustprogress nounwind
define internal void @__kmpc_target_deinit() #4 {
entry:
  %WorkFn = alloca ptr, align 8
  %0 = load i32, ptr addrspace(3) @IsSPMDMode, align 4, !tbaa !61
  %tobool.i.not = icmp eq i32 %0, 0
  br i1 %tobool.i.not, label %if.end, label %cleanup

if.end:                                           ; preds = %entry
  %1 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #18
  %sub.i.i = add i32 %1, -1
  %and.i.i = and i32 %sub.i.i, -32
  %2 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #18
  %cmp.i.i = icmp eq i32 %2, %and.i.i
  br i1 %cmp.i.i, label %if.then3, label %if.else

if.then3:                                         ; preds = %if.end
  store ptr null, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 40) to ptr addrspace(3)), align 8, !tbaa !76
  br label %cleanup

if.else:                                          ; preds = %if.end
  %3 = load ptr, ptr addrspace(3) @_ZL20KernelEnvironmentPtr, align 8, !tbaa !76
  %4 = load i8, ptr %3, align 8, !tbaa !91
  %tobool6.not = icmp eq i8 %4, 0
  br i1 %tobool6.not, label %if.then7, label %cleanup

if.then7:                                         ; preds = %if.else
  call void @llvm.lifetime.start.p0(i64 noundef 8, ptr noundef nonnull align 8 dereferenceable(8) %WorkFn) #29
  store ptr null, ptr %WorkFn, align 8, !tbaa !76
  %call8 = call zeroext i1 @__kmpc_kernel_parallel(ptr noalias nocapture nofree noundef nonnull writeonly align 8 dereferenceable(8) %WorkFn) #22
  %5 = load i32, ptr @__omp_rtl_debug_kind, align 4, !tbaa !61
  %6 = load i32, ptr addrspace(4) @__omp_rtl_device_environment, align 8, !tbaa !77
  %and.i.i1 = and i32 %5, 1
  %and.i = and i32 %and.i.i1, %6
  %tobool.i2.not = icmp eq i32 %and.i, 0
  %7 = load ptr, ptr %WorkFn, align 8
  %cmp = icmp eq ptr %7, null
  %or.cond = select i1 %tobool.i2.not, i1 true, i1 %cmp
  br i1 %or.cond, label %if.else11, label %if.then10

if.then10:                                        ; preds = %if.then7
  tail call fastcc void @__assert_fail_internal(ptr nofree noundef nonnull dereferenceable(18) @.str2, ptr noundef null, ptr nofree noundef nonnull dereferenceable(70) @.str15, i32 noundef 150, ptr nofree noundef nonnull dereferenceable(28) @__PRETTY_FUNCTION__.__kmpc_target_deinit) #20
  unreachable

if.else11:                                        ; preds = %if.then7
  tail call void @llvm.assume(i1 noundef %cmp) #21
  call void @llvm.lifetime.end.p0(i64 noundef 8, ptr noundef nonnull %WorkFn) #22
  br label %cleanup

cleanup:                                          ; preds = %if.else11, %if.else, %if.then3, %entry
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #5

attributes #0 = { convergent noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "omp_target_thread_limit"="128" "stack-protector-buffer-size"="8" "target-cpu"="sm_61" "target-features"="+ptx78,+sm_61" }
attributes #1 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_61" "target-features"="+ptx78,+sm_61" }
attributes #2 = { convergent mustprogress noinline norecurse nounwind optnone "frame-pointer"="all" "kernel" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_61" "target-features"="+ptx78,+sm_61" }
attributes #3 = { convergent noinline nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_61" "target-features"="+ptx78,+sm_61" }
attributes #4 = { convergent mustprogress nounwind "frame-pointer"="all" "llvm.assume"="ompx_no_call_asm" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_61" "target-features"="+ptx63,+ptx78,+sm_61" }
attributes #5 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #6 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "llvm.assume"="ompx_no_call_asm,ompx_aligned_barrier" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_61" "target-features"="+ptx63,+ptx78,+sm_61" }
attributes #7 = { convergent mustprogress noreturn nounwind "frame-pointer"="all" "llvm.assume"="ompx_no_call_asm" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_61" "target-features"="+ptx63,+ptx78,+sm_61" }
attributes #8 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #9 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #10 = { convergent nocallback nounwind }
attributes #11 = { convergent mustprogress nofree noinline norecurse nosync nounwind willreturn memory(read, argmem: write, inaccessiblemem: none) "frame-pointer"="all" "llvm.assume"="ompx_no_call_asm" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_61" "target-features"="+ptx63,+ptx78,+sm_61" }
attributes #12 = { convergent mustprogress noinline nounwind "frame-pointer"="all" "llvm.assume"="ompx_no_call_asm" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_61" "target-features"="+ptx63,+ptx78,+sm_61" }
attributes #13 = { convergent mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="all" "llvm.assume"="ompx_no_call_asm" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_61" "target-features"="+ptx63,+ptx78,+sm_61" }
attributes #14 = { convergent "frame-pointer"="all" "llvm.assume"="ompx_no_call_asm" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_61" "target-features"="+ptx63,+ptx78,+sm_61" }
attributes #15 = { cold noreturn nounwind memory(inaccessiblemem: write) }
attributes #16 = { convergent }
attributes #17 = { nounwind }
attributes #18 = { "llvm.assume"="ompx_no_call_asm" }
attributes #19 = { convergent nounwind "llvm.assume"="ompx_no_call_asm,ompx_aligned_barrier" }
attributes #20 = { noreturn nounwind "llvm.assume"="ompx_no_call_asm" }
attributes #21 = { memory(write) "llvm.assume"="ompx_no_call_asm" }
attributes #22 = { nounwind "llvm.assume"="ompx_no_call_asm" }
attributes #23 = { convergent nounwind }
attributes #24 = { convergent nounwind "llvm.assume"="ompx_no_call_asm" }
attributes #25 = { "llvm.assume"="ompx_no_call_asm,ompx_aligned_barrier" }
attributes #26 = { noreturn "llvm.assume"="ompx_no_call_asm" }
attributes #27 = { nofree willreturn "llvm.assume"="ompx_no_call_asm" }
attributes #28 = { convergent nounwind willreturn "llvm.assume"="ompx_no_call_asm" }
attributes #29 = { nofree nounwind willreturn "llvm.assume"="ompx_no_call_asm" }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9}
!llvm.dbg.cu = !{!10}
!nvvm.annotations = !{!12, !13}
!omp_offload.info = !{!14}
!llvm.ident = !{!15, !16, !15, !15, !15, !15, !15, !15, !15, !15, !15, !15, !15, !15, !15, !15}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 11, i32 8]}
!1 = !{i32 7, !"Dwarf Version", i32 2}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 7, !"openmp", i32 51}
!5 = !{i32 7, !"openmp-device", i32 51}
!6 = !{i32 8, !"PIC Level", i32 2}
!7 = !{i32 7, !"frame-pointer", i32 2}
!8 = !{i32 1, !"ThinLTO", i32 0}
!9 = !{i32 1, !"EnableSplitLTOUnit", i32 1}
!10 = distinct !DICompileUnit(language: DW_LANG_C11, file: !11, producer: "clang version 19.0.0git", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!11 = !DIFile(filename: "test.c", directory: "/tmp")
!12 = !{ptr @__omp_offloading_10305_5c00dd_h_l12_debug__, !"maxntidx", i32 128}
!13 = !{ptr @__omp_offloading_10305_5c00dd_h_l12, !"kernel", i32 1}
!14 = !{i32 0, i32 66309, i32 6029533, !"h", i32 12, i32 0, i32 0}
!15 = !{!"clang version 19.0.0git"}
!16 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!17 = distinct !DISubprogram(name: "__omp_offloading_10305_5c00dd_h_l12_debug__", scope: !11, file: !11, line: 13, type: !18, scopeLine: 13, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !10, retainedNodes: !23)
!18 = !DISubroutineType(types: !19)
!19 = !{null, !20}
!20 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !21)
!21 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !22)
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!23 = !{}
!24 = !DILocalVariable(name: "dyn_ptr", arg: 1, scope: !17, type: !20, flags: DIFlagArtificial)
!25 = !DILocation(line: 0, scope: !17)
!26 = !DILocation(line: 13, column: 3, scope: !17)
!27 = !DILocalVariable(name: "i", scope: !28, file: !11, line: 14, type: !29)
!28 = distinct !DILexicalBlock(scope: !17, file: !11, line: 13, column: 3)
!29 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!30 = !DILocation(line: 14, column: 9, scope: !28)
!31 = !DILocalVariable(name: "a", scope: !28, file: !11, line: 15, type: !32)
!32 = !DICompositeType(tag: DW_TAG_array_type, baseType: !29, size: 64, elements: !33)
!33 = !{!34}
!34 = !DISubrange(count: 2)
!35 = !DILocation(line: 15, column: 9, scope: !28)
!36 = !DILocation(line: 16, column: 5, scope: !28)
!37 = !DILocation(line: 17, column: 5, scope: !28)
!38 = !DILocation(line: 18, column: 3, scope: !28)
!39 = !DILocation(line: 18, column: 3, scope: !17)
!40 = distinct !DISubprogram(name: "__omp_offloading_10305_5c00dd_h_l12", scope: !11, file: !11, line: 12, type: !18, scopeLine: 12, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !10, retainedNodes: !23)
!41 = !DILocalVariable(name: "dyn_ptr", arg: 1, scope: !40, type: !20, flags: DIFlagArtificial)
!42 = !DILocation(line: 0, scope: !40)
!43 = !DILocation(line: 12, column: 1, scope: !40)
!44 = distinct !DISubprogram(name: "g", scope: !11, file: !11, line: 3, type: !45, scopeLine: 3, spFlags: DISPFlagDefinition, unit: !10, retainedNodes: !23)
!45 = !DISubroutineType(types: !46)
!46 = !{null}
!47 = !DILocalVariable(name: "i", scope: !44, file: !11, line: 4, type: !29)
!48 = !DILocation(line: 4, column: 7, scope: !44)
!49 = !DILocalVariable(name: "a", scope: !44, file: !11, line: 5, type: !32)
!50 = !DILocation(line: 5, column: 7, scope: !44)
!51 = !DILocation(line: 6, column: 3, scope: !44)
!52 = !DILocation(line: 7, column: 3, scope: !44)
!53 = !DILocation(line: 8, column: 1, scope: !44)
!54 = !{!55, !58, i64 2}
!55 = !{!"_ZTS26ConfigurationEnvironmentTy", !56, i64 0, !56, i64 1, !58, i64 2, !59, i64 4, !59, i64 8, !59, i64 12, !59, i64 16, !59, i64 20, !59, i64 24}
!56 = !{!"omnipotent char", !57, i64 0}
!57 = !{!"Simple C++ TBAA"}
!58 = !{!"_ZTSN4llvm3omp19OMPTgtExecModeFlagsE", !56, i64 0}
!59 = !{!"int", !56, i64 0}
!60 = !{!55, !56, i64 0}
!61 = !{!59, !59, i64 0}
!62 = !{!56, !56, i64 0}
!63 = !{!64, !59, i64 0}
!64 = !{!"_ZTSN4ompx5state11TeamStateTyE", !65, i64 0, !59, i64 28, !59, i64 32, !66, i64 40}
!65 = !{!"_ZTSN4ompx5state10ICVStateTyE", !59, i64 0, !59, i64 4, !59, i64 8, !59, i64 12, !59, i64 16, !59, i64 20, !59, i64 24}
!66 = !{!"any pointer", !56, i64 0}
!67 = !{!64, !59, i64 4}
!68 = !{!64, !59, i64 8}
!69 = !{!64, !59, i64 12}
!70 = !{!64, !59, i64 16}
!71 = !{!64, !59, i64 20}
!72 = !{!64, !59, i64 24}
!73 = !{!64, !59, i64 28}
!74 = !{!64, !59, i64 32}
!75 = !{!64, !66, i64 40}
!76 = !{!66, !66, i64 0}
!77 = !{!78, !59, i64 0}
!78 = !{!"_ZTS19DeviceEnvironmentTy", !59, i64 0, !59, i64 4, !59, i64 8, !59, i64 12, !79, i64 16, !79, i64 24, !79, i64 32, !79, i64 40}
!79 = !{!"long", !56, i64 0}
!80 = !{!65, !59, i64 0}
!81 = !{!65, !59, i64 4}
!82 = !{!65, !59, i64 8}
!83 = !{!65, !59, i64 16}
!84 = !{!65, !59, i64 20}
!85 = !{!65, !59, i64 24}
!86 = distinct !{!86, !87}
!87 = !{!"llvm.loop.mustprogress"}
!88 = !{!"branch_weights", i32 2000, i32 1}
!89 = !{!90, !66, i64 32}
!90 = !{!"_ZTSN4ompx5state13ThreadStateTyE", !65, i64 0, !66, i64 32}
!91 = !{!92, !56, i64 0}
!92 = !{!"_ZTS19KernelEnvironmentTy", !55, i64 0, !66, i64 32, !66, i64 40}
