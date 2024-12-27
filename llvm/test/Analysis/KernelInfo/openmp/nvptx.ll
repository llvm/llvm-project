; See ./README.md for how to maintain the LLVM IR in this test.

; REQUIRES: nvptx-registered-target

; RUN: opt -pass-remarks=kernel-info -passes=kernel-info \
; RUN:     -disable-output %s 2>&1 | \
; RUN:   FileCheck -match-full-lines %s

;  CHECK-NOT: remark:
;      CHECK: remark: test.c:0:0: in artificial function '[[OFF_FUNC:__omp_offloading_[a-f0-9_]*_h_l12]]_debug__', artificial alloca ('%[[#]]') for 'dyn_ptr' with static size of 8 bytes
; CHECK-NEXT: remark: test.c:14:9: in artificial function '[[OFF_FUNC]]_debug__', alloca ('%[[#]]') for 'i' with static size of 4 bytes
; CHECK-NEXT: remark: test.c:15:9: in artificial function '[[OFF_FUNC]]_debug__', alloca ('%[[#]]') for 'a' with static size of 8 bytes
; CHECK-NEXT: remark: <unknown>:0:0: in artificial function '[[OFF_FUNC]]_debug__', 'store' accesses memory in flat address space
; CHECK-NEXT: remark: test.c:13:3: in artificial function '[[OFF_FUNC]]_debug__', direct call to defined function, callee is '@__kmpc_target_init'
; CHECK-NEXT: remark: test.c:16:5: in artificial function '[[OFF_FUNC]]_debug__', direct call, callee is '@f'
; CHECK-NEXT: remark: test.c:17:5: in artificial function '[[OFF_FUNC]]_debug__', direct call to defined function, callee is 'g'
; CHECK-NEXT: remark: test.c:18:3: in artificial function '[[OFF_FUNC]]_debug__', direct call to defined function, callee is '@__kmpc_target_deinit'
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', ExternalNotKernel = 0
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', Allocas = 3
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', AllocasStaticSizeSum = 20
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', AllocasDyn = 0
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', DirectCalls = 4
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', IndirectCalls = 0
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', DirectCallsToDefinedFunctions = 3
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', InlineAssemblyCalls = 0
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', Invokes = 0
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', FlatAddrspaceAccesses = 1
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', FloatingPointOpProfileCount = 0

; CHECK-NEXT: remark: test.c:0:0: in artificial function '[[OFF_FUNC]]', artificial alloca ('%[[#]]') for 'dyn_ptr' with static size of 8 bytes
; CHECK-NEXT: remark: <unknown>:0:0: in artificial function '[[OFF_FUNC]]', 'store' accesses memory in flat address space
; CHECK-NEXT: remark: test.c:12:1: in artificial function '[[OFF_FUNC]]', 'load' ('%[[#]]') accesses memory in flat address space
; CHECK-NEXT: remark: test.c:12:1: in artificial function '[[OFF_FUNC]]', direct call to defined function, callee is artificial '[[OFF_FUNC]]_debug__'
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', ExternalNotKernel = 0
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', omp_target_thread_limit = 128
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', maxntidx = 128
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', Allocas = 1
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', AllocasStaticSizeSum = 8
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', AllocasDyn = 0
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', DirectCalls = 1
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', IndirectCalls = 0
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', DirectCallsToDefinedFunctions = 1
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', InlineAssemblyCalls = 0
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', Invokes = 0
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', FlatAddrspaceAccesses = 2
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', FloatingPointOpProfileCount = 0

; CHECK-NEXT: remark: test.c:4:7: in function 'g', alloca ('%[[#]]') for 'i' with static size of 4 bytes
; CHECK-NEXT: remark: test.c:5:7: in function 'g', alloca ('%[[#]]') for 'a' with static size of 8 bytes
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
; CHECK-NEXT: remark: test.c:3:0: in function 'g', FloatingPointOpProfileCount = 0
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
%"struct.rpc::Client" = type { %"struct.rpc::Process" }
%"struct.rpc::Process" = type { i32, ptr, ptr, ptr, ptr, [128 x i32] }
%"struct.(anonymous namespace)::SharedMemorySmartStackTy" = type { [512 x i8], [1024 x i8] }
%"struct.ompx::state::TeamStateTy" = type { %"struct.ompx::state::ICVStateTy", i32, i32, ptr }
%"struct.ompx::state::ICVStateTy" = type { i32, i32, i32, i32, i32, i32, i32 }
%printf_args = type { ptr, i32, ptr, ptr, ptr }
%printf_args.7 = type { ptr, i32, ptr, ptr }

@__omp_rtl_assume_teams_oversubscription = weak_odr hidden constant i32 0
@__omp_rtl_assume_threads_oversubscription = weak_odr hidden constant i32 0
@0 = private unnamed_addr constant [58 x i8] c";test.c;__omp_offloading_fd02_100102_h_l12_debug__;13;3;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 57, ptr @0 }, align 8
@__omp_offloading_fd02_100102_h_l12_dynamic_environment = weak_odr protected global %struct.DynamicEnvironmentTy zeroinitializer
@__omp_offloading_fd02_100102_h_l12_kernel_environment = weak_odr protected constant %struct.KernelEnvironmentTy { %struct.ConfigurationEnvironmentTy { i8 1, i8 1, i8 1, i32 1, i32 128, i32 -1, i32 -1, i32 0, i32 0 }, ptr @1, ptr @__omp_offloading_fd02_100102_h_l12_dynamic_environment }
@llvm.used = appending global [4 x ptr] [ptr @__llvm_rpc_client, ptr addrspacecast (ptr addrspace(4) @__omp_rtl_device_environment to ptr), ptr @__omp_rtl_device_memory_pool, ptr @__omp_rtl_device_memory_pool_tracker], section "llvm.metadata"
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
@__llvm_rpc_client = weak protected global %"struct.rpc::Client" zeroinitializer, align 8
@.str1125 = private unnamed_addr constant [48 x i8] c"/tmp/llvm/offload/DeviceRTL/src/Parallelism.cpp\00", align 1
@.str13 = private unnamed_addr constant [23 x i8] c"!mapping::isSPMDMode()\00", align 1
@__PRETTY_FUNCTION__.__kmpc_kernel_end_parallel = private unnamed_addr constant [34 x i8] c"void __kmpc_kernel_end_parallel()\00", align 1
@_ZL20KernelEnvironmentPtr = internal unnamed_addr addrspace(3) global ptr undef, align 8
@_ZL26KernelLaunchEnvironmentPtr = internal unnamed_addr addrspace(3) global ptr undef, align 8
@_ZN12_GLOBAL__N_122SharedMemorySmartStackE = internal addrspace(3) global %"struct.(anonymous namespace)::SharedMemorySmartStackTy" undef, align 16
@.str542 = private unnamed_addr constant [42 x i8] c"/tmp/llvm/offload/DeviceRTL/src/State.cpp\00", align 1
@.str845 = private unnamed_addr constant [33 x i8] c"NThreadsVar == Other.NThreadsVar\00", align 1
@__PRETTY_FUNCTION__._ZNK4ompx5state10ICVStateTy11assertEqualERKS1_ = private unnamed_addr constant [68 x i8] c"void ompx::state::ICVStateTy::assertEqual(const ICVStateTy &) const\00", align 1
@.str946 = private unnamed_addr constant [27 x i8] c"LevelVar == Other.LevelVar\00", align 1
@.str1047 = private unnamed_addr constant [39 x i8] c"ActiveLevelVar == Other.ActiveLevelVar\00", align 1
@.str1148 = private unnamed_addr constant [47 x i8] c"MaxActiveLevelsVar == Other.MaxActiveLevelsVar\00", align 1
@.str1249 = private unnamed_addr constant [33 x i8] c"RunSchedVar == Other.RunSchedVar\00", align 1
@.str1350 = private unnamed_addr constant [43 x i8] c"RunSchedChunkVar == Other.RunSchedChunkVar\00", align 1
@.str14 = private unnamed_addr constant [43 x i8] c"ParallelTeamSize == Other.ParallelTeamSize\00", align 1
@__PRETTY_FUNCTION__._ZNK4ompx5state11TeamStateTy11assertEqualERS1_ = private unnamed_addr constant [64 x i8] c"void ompx::state::TeamStateTy::assertEqual(TeamStateTy &) const\00", align 1
@.str1551 = private unnamed_addr constant [39 x i8] c"HasThreadState == Other.HasThreadState\00", align 1
@.str24 = private unnamed_addr constant [32 x i8] c"mapping::isSPMDMode() == IsSPMD\00", align 1
@__PRETTY_FUNCTION__._ZN4ompx5state18assumeInitialStateEb = private unnamed_addr constant [43 x i8] c"void ompx::state::assumeInitialState(bool)\00", align 1
@_ZL9ThreadDST = internal unnamed_addr addrspace(3) global ptr undef, align 8
@_ZN4ompx5state9TeamStateE = internal local_unnamed_addr addrspace(3) global %"struct.ompx::state::TeamStateTy" undef, align 8
@_ZN4ompx5state12ThreadStatesE = internal addrspace(3) global ptr undef, align 8

; Function Attrs: convergent noinline norecurse nounwind optnone
define internal void @__omp_offloading_fd02_100102_h_l12_debug__(ptr noalias noundef %0) #0 !dbg !19 {
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca [2 x i32], align 4
  store ptr %0, ptr %2, align 8
    #dbg_declare(ptr %2, !26, !DIExpression(), !27)
  %5 = call i32 @__kmpc_target_init(ptr @__omp_offloading_fd02_100102_h_l12_kernel_environment, ptr %0), !dbg !28
  %6 = icmp eq i32 %5, -1, !dbg !28
  br i1 %6, label %7, label %8, !dbg !28

7:                                                ; preds = %1
    #dbg_declare(ptr %3, !29, !DIExpression(), !32)
    #dbg_declare(ptr %4, !33, !DIExpression(), !37)
  call void @f() #16, !dbg !38
  call void @g() #16, !dbg !39
  call void @__kmpc_target_deinit(), !dbg !40
  ret void, !dbg !41

8:                                                ; preds = %1
  ret void, !dbg !28
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define weak_odr protected void @__omp_offloading_fd02_100102_h_l12(ptr noalias noundef %0) #1 !dbg !42 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
    #dbg_declare(ptr %2, !43, !DIExpression(), !44)
  %3 = load ptr, ptr %2, align 8, !dbg !45
  call void @__omp_offloading_fd02_100102_h_l12_debug__(ptr %3) #17, !dbg !45
  ret void, !dbg !45
}

; Function Attrs: convergent
declare void @f(...) #2

; Function Attrs: convergent noinline nounwind optnone
define hidden void @g() #3 !dbg !46 {
  %1 = alloca i32, align 4
  %2 = alloca [2 x i32], align 4
    #dbg_declare(ptr %1, !49, !DIExpression(), !50)
    #dbg_declare(ptr %2, !51, !DIExpression(), !52)
  call void @f() #16, !dbg !53
  call void @g() #16, !dbg !54
  ret void, !dbg !55
}

; Function Attrs: convergent mustprogress nounwind
define internal noundef range(i32 -1, 1024) i32 @__kmpc_target_init(ptr nofree noundef nonnull align 8 dereferenceable(48) %0, ptr nofree noundef nonnull align 8 dereferenceable(16) %1) #4 {
  %3 = alloca ptr, align 8
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 2
  %5 = load i8, ptr %4, align 2, !tbaa !56
  %6 = and i8 %5, 2
  %7 = icmp eq i8 %6, 0
  %8 = load i8, ptr %0, align 8, !tbaa !62
  %9 = icmp ne i8 %8, 0
  br i1 %7, label %21, label %10

10:                                               ; preds = %2
  %11 = tail call range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #18
  %12 = icmp eq i32 %11, 0
  br i1 %12, label %13, label %14

13:                                               ; preds = %10
  store i32 1, ptr addrspace(3) @IsSPMDMode, align 4, !tbaa !63
  store i8 0, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN12_GLOBAL__N_122SharedMemorySmartStackE to ptr), i64 512) to ptr addrspace(3)), align 1, !tbaa !64
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(48) addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i8 noundef 0, i64 noundef 16, i1 noundef false) #18
  store i32 1, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 16) to ptr addrspace(3)), align 8, !tbaa !65
  store i32 1, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 20) to ptr addrspace(3)), align 4, !tbaa !70
  store i32 1, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 24) to ptr addrspace(3)), align 8, !tbaa !71
  store i32 1, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 28) to ptr addrspace(3)), align 4, !tbaa !72
  store i32 0, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 32) to ptr addrspace(3)), align 8, !tbaa !73
  store ptr null, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 40) to ptr addrspace(3)), align 8, !tbaa !74
  store ptr null, ptr addrspace(3) @_ZN4ompx5state12ThreadStatesE, align 8, !tbaa !75
  store ptr %0, ptr addrspace(3) @_ZL20KernelEnvironmentPtr, align 8, !tbaa !77
  store ptr %1, ptr addrspace(3) @_ZL26KernelLaunchEnvironmentPtr, align 8, !tbaa !79
  br label %18

14:                                               ; preds = %10
  %15 = zext nneg i32 %11 to i64
  %16 = getelementptr inbounds nuw [1024 x i8], ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN12_GLOBAL__N_122SharedMemorySmartStackE to ptr), i64 512), i64 0, i64 %15
  %17 = addrspacecast ptr %16 to ptr addrspace(3)
  store i8 0, ptr addrspace(3) %17, align 1, !tbaa !64
  br label %18

18:                                               ; preds = %14, %13
  br i1 %12, label %19, label %20

19:                                               ; preds = %18
  store ptr null, ptr addrspace(3) @_ZL9ThreadDST, align 8, !tbaa !81
  br label %20

20:                                               ; preds = %18, %19
  tail call void @_ZN4ompx11synchronize14threadsAlignedENS_6atomic10OrderingTyE(i32 poison) #19
  br label %37

21:                                               ; preds = %2
  %22 = tail call range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #18, !range !83
  %23 = add nsw i32 %22, -1
  %24 = and i32 %23, -32
  %25 = tail call range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #18
  %26 = icmp eq i32 %25, %24
  br i1 %26, label %27, label %31

27:                                               ; preds = %21
  store i32 0, ptr addrspace(3) @IsSPMDMode, align 4, !tbaa !63
  %28 = zext nneg i32 %25 to i64
  %29 = getelementptr inbounds nuw [1024 x i8], ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN12_GLOBAL__N_122SharedMemorySmartStackE to ptr), i64 512), i64 0, i64 %28
  %30 = addrspacecast ptr %29 to ptr addrspace(3)
  store i8 0, ptr addrspace(3) %30, align 1, !tbaa !64
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(48) addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i8 noundef 0, i64 noundef 16, i1 noundef false) #18
  store i32 1, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 16) to ptr addrspace(3)), align 8, !tbaa !65
  store i32 1, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 20) to ptr addrspace(3)), align 4, !tbaa !70
  store i32 1, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 24) to ptr addrspace(3)), align 8, !tbaa !71
  store i32 1, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 28) to ptr addrspace(3)), align 4, !tbaa !72
  store i32 0, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 32) to ptr addrspace(3)), align 8, !tbaa !73
  store ptr null, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 40) to ptr addrspace(3)), align 8, !tbaa !74
  store ptr null, ptr addrspace(3) @_ZN4ompx5state12ThreadStatesE, align 8, !tbaa !75
  store ptr %0, ptr addrspace(3) @_ZL20KernelEnvironmentPtr, align 8, !tbaa !77
  store ptr %1, ptr addrspace(3) @_ZL26KernelLaunchEnvironmentPtr, align 8, !tbaa !79
  br label %35

31:                                               ; preds = %21
  %32 = zext nneg i32 %25 to i64
  %33 = getelementptr inbounds nuw [1024 x i8], ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN12_GLOBAL__N_122SharedMemorySmartStackE to ptr), i64 512), i64 0, i64 %32
  %34 = addrspacecast ptr %33 to ptr addrspace(3)
  store i8 0, ptr addrspace(3) %34, align 1, !tbaa !64
  br label %35

35:                                               ; preds = %31, %27
  br i1 %26, label %36, label %37

36:                                               ; preds = %35
  store ptr null, ptr addrspace(3) @_ZL9ThreadDST, align 8, !tbaa !81
  br label %37

37:                                               ; preds = %36, %35, %20
  br i1 %7, label %100, label %38

38:                                               ; preds = %37
  %39 = load i32, ptr @__omp_rtl_debug_kind, align 4, !tbaa !63
  %40 = load i32, ptr addrspace(4) @__omp_rtl_device_environment, align 8, !tbaa !84
  %41 = and i32 %39, 1
  %42 = and i32 %41, %40
  %43 = icmp ne i32 %42, 0
  %44 = load i32, ptr addrspace(3) @_ZN4ompx5state9TeamStateE, align 8, !tbaa !87
  %45 = icmp ne i32 %44, 0
  %46 = select i1 %43, i1 %45, i1 false
  br i1 %46, label %47, label %48

47:                                               ; preds = %38
  tail call fastcc void @__assert_fail_internal(ptr nofree noundef nonnull dereferenceable(33) @.str845, ptr noundef null, ptr nofree noundef nonnull dereferenceable(66) @.str542, i32 noundef 193, ptr nofree noundef nonnull dereferenceable(68) @__PRETTY_FUNCTION__._ZNK4ompx5state10ICVStateTy11assertEqualERKS1_) #20
  unreachable

48:                                               ; preds = %38
  %49 = icmp eq i32 %44, 0
  tail call void @llvm.assume(i1 noundef %49) #21
  %50 = load i32, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 4) to ptr addrspace(3)), align 4, !tbaa !88
  br i1 %43, label %51, label %54

51:                                               ; preds = %48
  %52 = icmp eq i32 %50, 0
  br i1 %52, label %54, label %53

53:                                               ; preds = %51
  tail call fastcc void @__assert_fail_internal(ptr nofree noundef nonnull dereferenceable(27) @.str946, ptr noundef null, ptr nofree noundef nonnull dereferenceable(66) @.str542, i32 noundef 194, ptr nofree noundef nonnull dereferenceable(68) @__PRETTY_FUNCTION__._ZNK4ompx5state10ICVStateTy11assertEqualERKS1_) #20
  unreachable

54:                                               ; preds = %51, %48
  %55 = phi i32 [ 0, %51 ], [ %50, %48 ]
  %56 = icmp eq i32 %55, 0
  tail call void @llvm.assume(i1 noundef %56) #21
  %57 = load i32, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 8) to ptr addrspace(3)), align 8, !tbaa !89
  br i1 %43, label %58, label %61

58:                                               ; preds = %54
  %59 = icmp eq i32 %57, 0
  br i1 %59, label %61, label %60

60:                                               ; preds = %58
  tail call fastcc void @__assert_fail_internal(ptr nofree noundef nonnull dereferenceable(39) @.str1047, ptr noundef null, ptr nofree noundef nonnull dereferenceable(66) @.str542, i32 noundef 195, ptr nofree noundef nonnull dereferenceable(68) @__PRETTY_FUNCTION__._ZNK4ompx5state10ICVStateTy11assertEqualERKS1_) #20
  unreachable

61:                                               ; preds = %58, %54
  %62 = phi i32 [ 0, %58 ], [ %57, %54 ]
  %63 = icmp eq i32 %62, 0
  tail call void @llvm.assume(i1 noundef %63) #21
  %64 = load i32, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 16) to ptr addrspace(3)), align 8, !tbaa !90
  br i1 %43, label %65, label %68

65:                                               ; preds = %61
  %66 = icmp eq i32 %64, 1
  br i1 %66, label %68, label %67

67:                                               ; preds = %65
  tail call fastcc void @__assert_fail_internal(ptr nofree noundef nonnull dereferenceable(47) @.str1148, ptr noundef null, ptr nofree noundef nonnull dereferenceable(66) @.str542, i32 noundef 196, ptr nofree noundef nonnull dereferenceable(68) @__PRETTY_FUNCTION__._ZNK4ompx5state10ICVStateTy11assertEqualERKS1_) #20
  unreachable

68:                                               ; preds = %65, %61
  %69 = phi i32 [ 1, %65 ], [ %64, %61 ]
  %70 = icmp eq i32 %69, 1
  tail call void @llvm.assume(i1 noundef %70) #21
  %71 = load i32, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 20) to ptr addrspace(3)), align 4, !tbaa !91
  br i1 %43, label %72, label %93

72:                                               ; preds = %68
  %73 = icmp eq i32 %71, 1
  br i1 %73, label %75, label %74

74:                                               ; preds = %72
  tail call fastcc void @__assert_fail_internal(ptr nofree noundef nonnull dereferenceable(33) @.str1249, ptr noundef null, ptr nofree noundef nonnull dereferenceable(66) @.str542, i32 noundef 197, ptr nofree noundef nonnull dereferenceable(68) @__PRETTY_FUNCTION__._ZNK4ompx5state10ICVStateTy11assertEqualERKS1_) #20
  unreachable

75:                                               ; preds = %72
  %76 = icmp eq i32 1, 1
  tail call void @llvm.assume(i1 noundef %76) #21
  br i1 %43, label %77, label %95

77:                                               ; preds = %75
  %78 = load i32, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 24) to ptr addrspace(3)), align 8, !tbaa !92
  %79 = icmp eq i32 %78, 1
  br i1 %79, label %81, label %80

80:                                               ; preds = %77
  tail call fastcc void @__assert_fail_internal(ptr nofree noundef nonnull dereferenceable(43) @.str1350, ptr noundef null, ptr nofree noundef nonnull dereferenceable(66) @.str542, i32 noundef 198, ptr nofree noundef nonnull dereferenceable(68) @__PRETTY_FUNCTION__._ZNK4ompx5state10ICVStateTy11assertEqualERKS1_) #20
  unreachable

81:                                               ; preds = %77
  %82 = load i32, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 28) to ptr addrspace(3)), align 4, !tbaa !72
  %83 = icmp eq i32 %82, 1
  br i1 %83, label %85, label %84

84:                                               ; preds = %81
  tail call fastcc void @__assert_fail_internal(ptr nofree noundef nonnull dereferenceable(43) @.str14, ptr noundef null, ptr nofree noundef nonnull dereferenceable(66) @.str542, i32 noundef 222, ptr nofree noundef nonnull dereferenceable(64) @__PRETTY_FUNCTION__._ZNK4ompx5state11TeamStateTy11assertEqualERS1_) #20
  unreachable

85:                                               ; preds = %81
  %86 = load i32, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 32) to ptr addrspace(3)), align 8, !tbaa !73
  %87 = icmp eq i32 %86, 0
  br i1 %87, label %89, label %88

88:                                               ; preds = %85
  tail call fastcc void @__assert_fail_internal(ptr nofree noundef nonnull dereferenceable(39) @.str1551, ptr noundef null, ptr nofree noundef nonnull dereferenceable(66) @.str542, i32 noundef 223, ptr nofree noundef nonnull dereferenceable(64) @__PRETTY_FUNCTION__._ZNK4ompx5state11TeamStateTy11assertEqualERS1_) #20
  unreachable

89:                                               ; preds = %85
  %90 = load i32, ptr addrspace(3) @IsSPMDMode, align 4, !tbaa !63
  %91 = icmp eq i32 %90, 0
  br i1 %91, label %92, label %98

92:                                               ; preds = %89
  tail call fastcc void @__assert_fail_internal(ptr nofree noundef nonnull dereferenceable(32) @.str24, ptr noundef null, ptr nofree noundef nonnull dereferenceable(66) @.str542, i32 noundef 326, ptr nofree noundef nonnull dereferenceable(43) @__PRETTY_FUNCTION__._ZN4ompx5state18assumeInitialStateEb) #20
  unreachable

93:                                               ; preds = %68
  %94 = icmp eq i32 %71, 1
  tail call void @llvm.assume(i1 noundef %94) #21
  br label %95

95:                                               ; preds = %75, %93
  %96 = load i32, ptr addrspace(3) @IsSPMDMode, align 4, !tbaa !63
  %97 = icmp ne i32 %96, 0
  br label %98

98:                                               ; preds = %89, %95
  %99 = phi i1 [ %97, %95 ], [ true, %89 ]
  tail call void @llvm.assume(i1 noundef %99) #21
  tail call void @_ZN4ompx11synchronize14threadsAlignedENS_6atomic10OrderingTyE(i32 poison) #19
  br label %130

100:                                              ; preds = %37
  %101 = tail call range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #18, !range !83
  %102 = add nsw i32 %101, -1
  %103 = and i32 %102, -32
  %104 = tail call range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #18, !range !93
  %105 = icmp eq i32 %104, %103
  br i1 %105, label %130, label %106

106:                                              ; preds = %100
  %107 = add nsw i32 %101, -32
  %108 = icmp ult i32 %104, %107
  %109 = select i1 %9, i1 %108, i1 false
  br i1 %109, label %110, label %130

110:                                              ; preds = %106
  %111 = load i32, ptr @__omp_rtl_debug_kind, align 4
  %112 = load i32, ptr addrspace(4) @__omp_rtl_device_environment, align 8
  %113 = and i32 %111, 1
  %114 = and i32 %113, %112
  %115 = icmp ne i32 %114, 0
  br label %116

116:                                              ; preds = %110, %128
  call void @llvm.lifetime.start.p0(i64 noundef 8, ptr noundef nonnull align 8 dereferenceable(8) %3) #22
  store ptr null, ptr %3, align 8, !tbaa !94
  tail call void @llvm.nvvm.barrier.sync(i32 noundef 8) #18
  %117 = call zeroext i1 @__kmpc_kernel_parallel(ptr noalias nocapture nofree noundef nonnull writeonly align 8 dereferenceable(8) %3) #22
  %118 = load ptr, ptr %3, align 8, !tbaa !94
  %119 = icmp eq ptr %118, null
  br i1 %119, label %129, label %120

120:                                              ; preds = %116
  br i1 %117, label %121, label %128

121:                                              ; preds = %120
  %122 = load i32, ptr addrspace(3) @IsSPMDMode, align 4
  %123 = icmp ne i32 %122, 0
  %124 = select i1 %115, i1 %123, i1 false
  br i1 %124, label %125, label %126

125:                                              ; preds = %121
  tail call fastcc void @__assert_fail_internal(ptr nofree noundef nonnull dereferenceable(23) @.str13, ptr noundef null, ptr nofree noundef nonnull dereferenceable(67) @.str15, i32 noundef 60, ptr nofree noundef nonnull dereferenceable(36) @__PRETTY_FUNCTION__._ZL19genericStateMachineP7IdentTy) #20
  unreachable

126:                                              ; preds = %121
  %127 = icmp eq i32 %122, 0
  tail call void @llvm.assume(i1 noundef %127) #21
  tail call void %118(i32 noundef 0, i32 noundef %104) #23
  tail call void @__kmpc_kernel_end_parallel() #24
  br label %128

128:                                              ; preds = %126, %120
  tail call void @llvm.nvvm.barrier.sync(i32 noundef 8) #18
  call void @llvm.lifetime.end.p0(i64 noundef 8, ptr noundef nonnull %3) #22
  br label %116, !llvm.loop !95

129:                                              ; preds = %116
  call void @llvm.lifetime.end.p0(i64 noundef 8, ptr noundef nonnull %3) #22
  br label %130

130:                                              ; preds = %106, %129, %100, %98
  %131 = phi i32 [ -1, %98 ], [ -1, %100 ], [ %104, %129 ], [ %104, %106 ]
  ret i32 %131
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.tid.x() #5

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: write)
declare void @llvm.memset.p0.i64(ptr nocapture writeonly, i8, i64, i1 immarg) #6

; Function Attrs: convergent mustprogress noinline norecurse nounwind
define internal void @_ZN4ompx11synchronize14threadsAlignedENS_6atomic10OrderingTyE(i32 %0) local_unnamed_addr #7 {
  tail call void @llvm.nvvm.barrier0() #25
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #5

; Function Attrs: cold convergent mustprogress noreturn nounwind
define internal fastcc void @__assert_fail_internal(ptr nofree noundef nonnull dereferenceable(8) %0, ptr noundef %1, ptr nofree noundef nonnull dereferenceable(66) %2, i32 noundef range(i32 60, 905) %3, ptr nofree noundef nonnull dereferenceable(20) %4) unnamed_addr #8 {
  %6 = alloca %printf_args, align 8
  %7 = alloca %printf_args.7, align 8
  %8 = icmp eq ptr %1, null
  br i1 %8, label %12, label %9

9:                                                ; preds = %5
  store ptr %2, ptr %6, align 8
  %10 = getelementptr inbounds nuw i8, ptr %6, i64 8
  store i32 %3, ptr %10, align 8
  %11 = getelementptr inbounds nuw i8, ptr %6, i64 16
  store ptr %4, ptr %11, align 8
  br label %14

12:                                               ; preds = %5
  store ptr %2, ptr %7, align 8
  %13 = getelementptr inbounds nuw i8, ptr %7, i64 8
  store i32 %3, ptr %13, align 8
  br label %14

14:                                               ; preds = %12, %9
  %15 = phi i64 [ 16, %12 ], [ 24, %9 ]
  %16 = phi ptr [ %7, %12 ], [ %6, %9 ]
  %17 = phi ptr [ %4, %12 ], [ %1, %9 ]
  %18 = phi i64 [ 24, %12 ], [ 32, %9 ]
  %19 = phi ptr [ @.str1, %12 ], [ @.str, %9 ]
  %20 = getelementptr inbounds nuw i8, ptr %16, i64 %15
  store ptr %17, ptr %20, align 8
  %21 = getelementptr inbounds nuw i8, ptr %16, i64 %18
  store ptr %0, ptr %21, align 8
  %22 = call i32 @vprintf(ptr noundef nonnull %19, ptr noundef nonnull %16) #22
  call void @llvm.trap() #26
  unreachable
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #9

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.start.p0(i64 immarg, ptr nocapture) #10

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier.sync(i32) #11

; Function Attrs: convergent mustprogress nofree noinline norecurse nosync nounwind willreturn memory(read, argmem: write, inaccessiblemem: none)
define internal noundef zeroext i1 @__kmpc_kernel_parallel(ptr nocapture nofree noundef nonnull writeonly align 8 dereferenceable(8) initializes((0, 8)) %0) local_unnamed_addr #12 {
  %2 = load ptr, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 40) to ptr addrspace(3)), align 8, !tbaa !94
  store ptr %2, ptr %0, align 8, !tbaa !94
  %3 = icmp eq ptr %2, null
  br i1 %3, label %15, label %4

4:                                                ; preds = %1
  %5 = tail call noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #27, !range !93
  %6 = load i32, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 28) to ptr addrspace(3)), align 4, !tbaa !63
  %7 = icmp eq i32 %6, 0
  %8 = tail call range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #18, !range !83
  %9 = load i32, ptr addrspace(3) @IsSPMDMode, align 4
  %10 = icmp eq i32 %9, 0
  %11 = select i1 %10, i32 -32, i32 0
  %12 = add nsw i32 %11, %8
  %13 = select i1 %7, i32 %12, i32 %6
  %14 = icmp ult i32 %5, %13
  br label %15

15:                                               ; preds = %4, %1
  %16 = phi i1 [ %14, %4 ], [ false, %1 ]
  ret i1 %16
}

; Function Attrs: convergent mustprogress noinline nounwind
define internal void @__kmpc_kernel_end_parallel() local_unnamed_addr #13 {
  %1 = load i32, ptr @__omp_rtl_debug_kind, align 4, !tbaa !63
  %2 = load i32, ptr addrspace(4) @__omp_rtl_device_environment, align 8, !tbaa !84
  %3 = and i32 %1, 1
  %4 = and i32 %3, %2
  %5 = icmp ne i32 %4, 0
  %6 = load i32, ptr addrspace(3) @IsSPMDMode, align 4
  %7 = icmp ne i32 %6, 0
  %8 = select i1 %5, i1 %7, i1 false
  br i1 %8, label %9, label %10

9:                                                ; preds = %0
  tail call fastcc void @__assert_fail_internal(ptr nofree noundef nonnull dereferenceable(23) @.str13, ptr noundef null, ptr nofree noundef nonnull dereferenceable(72) @.str1125, i32 noundef 298, ptr nofree noundef nonnull dereferenceable(34) @__PRETTY_FUNCTION__.__kmpc_kernel_end_parallel) #20
  unreachable

10:                                               ; preds = %0
  %11 = icmp eq i32 %6, 0
  tail call void @llvm.assume(i1 noundef %11) #21
  %12 = load i32, ptr @__omp_rtl_assume_no_thread_state, align 4, !tbaa !63
  %13 = icmp eq i32 %12, 0
  %14 = load i32, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 32) to ptr addrspace(3)), align 8
  %15 = icmp ne i32 %14, 0
  %16 = select i1 %13, i1 %15, i1 false
  br i1 %16, label %17, label %30

17:                                               ; preds = %10
  %18 = tail call noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #27, !range !93
  %19 = load ptr, ptr addrspace(3) @_ZN4ompx5state12ThreadStatesE, align 8, !tbaa !75
  %20 = zext nneg i32 %18 to i64
  %21 = getelementptr inbounds nuw ptr, ptr %19, i64 %20
  %22 = load ptr, ptr %21, align 8, !tbaa !97
  %23 = icmp eq ptr %22, null
  br i1 %23, label %30, label %24, !prof !99

24:                                               ; preds = %17
  %25 = getelementptr inbounds nuw i8, ptr %22, i64 32
  %26 = load ptr, ptr %25, align 8, !tbaa !100
  tail call void @free(ptr noundef nonnull dereferenceable(40) %22) #28
  %27 = load ptr, ptr addrspace(3) @_ZN4ompx5state12ThreadStatesE, align 8, !tbaa !75
  %28 = getelementptr inbounds nuw ptr, ptr %27, i64 %20
  store ptr %26, ptr %28, align 8, !tbaa !97
  %29 = load i32, ptr addrspace(3) @IsSPMDMode, align 4
  br label %30

30:                                               ; preds = %10, %17, %24
  %31 = phi i32 [ 0, %10 ], [ 0, %17 ], [ %29, %24 ]
  %32 = icmp ne i32 %31, 0
  %33 = select i1 %5, i1 %32, i1 false
  br i1 %33, label %34, label %35

34:                                               ; preds = %30
  tail call fastcc void @__assert_fail_internal(ptr nofree noundef nonnull dereferenceable(23) @.str13, ptr noundef null, ptr nofree noundef nonnull dereferenceable(72) @.str1125, i32 noundef 301, ptr nofree noundef nonnull dereferenceable(34) @__PRETTY_FUNCTION__.__kmpc_kernel_end_parallel) #20
  unreachable

35:                                               ; preds = %30
  %36 = icmp eq i32 %31, 0
  tail call void @llvm.assume(i1 noundef %36) #21
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #10

; Function Attrs: convergent mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite)
declare extern_weak void @free(ptr allocptr nocapture noundef) local_unnamed_addr #14

; Function Attrs: convergent
declare i32 @vprintf(ptr, ptr) local_unnamed_addr #2

; Function Attrs: cold noreturn nounwind memory(inaccessiblemem: write)
declare void @llvm.trap() #15

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier0() #11

; Function Attrs: convergent mustprogress nounwind
define internal void @__kmpc_target_deinit() #4 {
  %1 = alloca ptr, align 8
  %2 = load i32, ptr addrspace(3) @IsSPMDMode, align 4, !tbaa !63
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %4, label %27

4:                                                ; preds = %0
  %5 = tail call range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #18, !range !83
  %6 = add nsw i32 %5, -1
  %7 = and i32 %6, -32
  %8 = tail call range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #18, !range !93
  %9 = icmp eq i32 %8, %7
  br i1 %9, label %10, label %11

10:                                               ; preds = %4
  store ptr null, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 40) to ptr addrspace(3)), align 8, !tbaa !94
  br label %27

11:                                               ; preds = %4
  %12 = load ptr, ptr addrspace(3) @_ZL20KernelEnvironmentPtr, align 8, !tbaa !77
  %13 = load i8, ptr %12, align 8, !tbaa !102
  %14 = icmp eq i8 %13, 0
  br i1 %14, label %15, label %27

15:                                               ; preds = %11
  call void @llvm.lifetime.start.p0(i64 noundef 8, ptr noundef nonnull align 8 dereferenceable(8) %1) #29
  store ptr null, ptr %1, align 8, !tbaa !94
  %16 = call zeroext i1 @__kmpc_kernel_parallel(ptr noalias nocapture nofree noundef nonnull writeonly align 8 dereferenceable(8) %1) #22
  %17 = load i32, ptr @__omp_rtl_debug_kind, align 4, !tbaa !63
  %18 = load i32, ptr addrspace(4) @__omp_rtl_device_environment, align 8, !tbaa !84
  %19 = and i32 %17, 1
  %20 = and i32 %19, %18
  %21 = icmp eq i32 %20, 0
  %22 = load ptr, ptr %1, align 8
  %23 = icmp eq ptr %22, null
  %24 = select i1 %21, i1 true, i1 %23
  br i1 %24, label %26, label %25

25:                                               ; preds = %15
  tail call fastcc void @__assert_fail_internal(ptr nofree noundef nonnull dereferenceable(18) @.str2, ptr noundef null, ptr nofree noundef nonnull dereferenceable(67) @.str15, i32 noundef 152, ptr nofree noundef nonnull dereferenceable(28) @__PRETTY_FUNCTION__.__kmpc_target_deinit) #20
  unreachable

26:                                               ; preds = %15
  tail call void @llvm.assume(i1 noundef %23) #21
  call void @llvm.lifetime.end.p0(i64 noundef 8, ptr noundef nonnull %1) #22
  br label %27

27:                                               ; preds = %26, %11, %10, %0
  ret void
}

attributes #0 = { convergent noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_70" "target-features"="+ptx83,+sm_70" }
attributes #1 = { convergent mustprogress noinline norecurse nounwind optnone "frame-pointer"="all" "kernel" "no-trapping-math"="true" "omp_target_thread_limit"="128" "stack-protector-buffer-size"="8" "target-cpu"="sm_70" "target-features"="+ptx83,+sm_70" }
attributes #2 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_70" "target-features"="+ptx83,+sm_70" }
attributes #3 = { convergent noinline nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_70" "target-features"="+ptx83,+sm_70" }
attributes #4 = { convergent mustprogress nounwind "frame-pointer"="all" "llvm.assume"="ompx_no_call_asm" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_70" "target-features"="+ptx63,+ptx83,+sm_70" }
attributes #5 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #6 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #7 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "llvm.assume"="ompx_aligned_barrier,ompx_no_call_asm" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_70" "target-features"="+ptx63,+ptx83,+sm_70" }
attributes #8 = { cold convergent mustprogress noreturn nounwind "frame-pointer"="all" "llvm.assume"="ompx_no_call_asm" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_70" "target-features"="+ptx63,+ptx83,+sm_70" }
attributes #9 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #10 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #11 = { convergent nocallback nounwind }
attributes #12 = { convergent mustprogress nofree noinline norecurse nosync nounwind willreturn memory(read, argmem: write, inaccessiblemem: none) "frame-pointer"="all" "llvm.assume"="ompx_no_call_asm" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_70" "target-features"="+ptx63,+ptx83,+sm_70" }
attributes #13 = { convergent mustprogress noinline nounwind "frame-pointer"="all" "llvm.assume"="ompx_no_call_asm" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_70" "target-features"="+ptx63,+ptx83,+sm_70" }
attributes #14 = { convergent mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="all" "llvm.assume"="ompx_no_call_asm" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_70" "target-features"="+ptx63,+ptx83,+sm_70" }
attributes #15 = { cold noreturn nounwind memory(inaccessiblemem: write) }
attributes #16 = { convergent }
attributes #17 = { nounwind }
attributes #18 = { "llvm.assume"="ompx_no_call_asm" }
attributes #19 = { convergent nounwind "llvm.assume"="ompx_aligned_barrier,ompx_no_call_asm" }
attributes #20 = { noreturn nounwind "llvm.assume"="ompx_no_call_asm" }
attributes #21 = { memory(write) "llvm.assume"="ompx_no_call_asm" }
attributes #22 = { nounwind "llvm.assume"="ompx_no_call_asm" }
attributes #23 = { convergent nounwind }
attributes #24 = { convergent nounwind "llvm.assume"="ompx_no_call_asm" }
attributes #25 = { "llvm.assume"="ompx_aligned_barrier,ompx_no_call_asm" }
attributes #26 = { noreturn "llvm.assume"="ompx_no_call_asm" }
attributes #27 = { nofree willreturn "llvm.assume"="ompx_no_call_asm" }
attributes #28 = { convergent nounwind willreturn "llvm.assume"="ompx_no_call_asm" }
attributes #29 = { nofree nounwind willreturn "llvm.assume"="ompx_no_call_asm" }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10}
!llvm.dbg.cu = !{!11}
!nvvm.annotations = !{!13, !14}
!omp_offload.info = !{!15}
!llvm.ident = !{!16, !17, !16, !16, !16, !16, !16, !16, !16, !16, !16, !16, !16, !16, !16, !16, !16}
!nvvmir.version = !{!18}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 12, i32 3]}
!1 = !{i32 7, !"Dwarf Version", i32 2}
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 1, !"wchar_size", i32 4}
!4 = !{i32 4, !"nvvm-reflect-ftz", i32 0}
!5 = !{i32 7, !"openmp", i32 51}
!6 = !{i32 7, !"openmp-device", i32 51}
!7 = !{i32 8, !"PIC Level", i32 2}
!8 = !{i32 7, !"frame-pointer", i32 2}
!9 = !{i32 1, !"ThinLTO", i32 0}
!10 = !{i32 1, !"EnableSplitLTOUnit", i32 1}
!11 = distinct !DICompileUnit(language: DW_LANG_C11, file: !12, producer: "clang version 20.0.0git (/tmp/llvm/clang 8982f8ff551bd4c11d47afefe97364c3a5c25ec8)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!12 = !DIFile(filename: "test.c", directory: "/tmp")
!13 = !{ptr @__omp_offloading_fd02_100102_h_l12, !"maxntidx", i32 128}
!14 = !{ptr @__omp_offloading_fd02_100102_h_l12, !"kernel", i32 1}
!15 = !{i32 0, i32 64770, i32 1048834, !"h", i32 12, i32 0, i32 0}
!16 = !{!"clang version 20.0.0git (/tmp/llvm/clang 8982f8ff551bd4c11d47afefe97364c3a5c25ec8)"}
!17 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!18 = !{i32 2, i32 0}
!19 = distinct !DISubprogram(name: "__omp_offloading_fd02_100102_h_l12_debug__", scope: !12, file: !12, line: 13, type: !20, scopeLine: 13, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !11, retainedNodes: !25)
!20 = !DISubroutineType(types: !21)
!21 = !{null, !22}
!22 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !23)
!23 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !24)
!24 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!25 = !{}
!26 = !DILocalVariable(name: "dyn_ptr", arg: 1, scope: !19, type: !22, flags: DIFlagArtificial)
!27 = !DILocation(line: 0, scope: !19)
!28 = !DILocation(line: 13, column: 3, scope: !19)
!29 = !DILocalVariable(name: "i", scope: !30, file: !12, line: 14, type: !31)
!30 = distinct !DILexicalBlock(scope: !19, file: !12, line: 13, column: 3)
!31 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!32 = !DILocation(line: 14, column: 9, scope: !30)
!33 = !DILocalVariable(name: "a", scope: !30, file: !12, line: 15, type: !34)
!34 = !DICompositeType(tag: DW_TAG_array_type, baseType: !31, size: 64, elements: !35)
!35 = !{!36}
!36 = !DISubrange(count: 2)
!37 = !DILocation(line: 15, column: 9, scope: !30)
!38 = !DILocation(line: 16, column: 5, scope: !30)
!39 = !DILocation(line: 17, column: 5, scope: !30)
!40 = !DILocation(line: 18, column: 3, scope: !30)
!41 = !DILocation(line: 18, column: 3, scope: !19)
!42 = distinct !DISubprogram(name: "__omp_offloading_fd02_100102_h_l12", scope: !12, file: !12, line: 12, type: !20, scopeLine: 12, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !11, retainedNodes: !25)
!43 = !DILocalVariable(name: "dyn_ptr", arg: 1, scope: !42, type: !22, flags: DIFlagArtificial)
!44 = !DILocation(line: 0, scope: !42)
!45 = !DILocation(line: 12, column: 1, scope: !42)
!46 = distinct !DISubprogram(name: "g", scope: !12, file: !12, line: 3, type: !47, scopeLine: 3, spFlags: DISPFlagDefinition, unit: !11, retainedNodes: !25)
!47 = !DISubroutineType(types: !48)
!48 = !{null}
!49 = !DILocalVariable(name: "i", scope: !46, file: !12, line: 4, type: !31)
!50 = !DILocation(line: 4, column: 7, scope: !46)
!51 = !DILocalVariable(name: "a", scope: !46, file: !12, line: 5, type: !34)
!52 = !DILocation(line: 5, column: 7, scope: !46)
!53 = !DILocation(line: 6, column: 3, scope: !46)
!54 = !DILocation(line: 7, column: 3, scope: !46)
!55 = !DILocation(line: 8, column: 1, scope: !46)
!56 = !{!57, !60, i64 2}
!57 = !{!"_ZTS26ConfigurationEnvironmentTy", !58, i64 0, !58, i64 1, !60, i64 2, !61, i64 4, !61, i64 8, !61, i64 12, !61, i64 16, !61, i64 20, !61, i64 24}
!58 = !{!"omnipotent char", !59, i64 0}
!59 = !{!"Simple C++ TBAA"}
!60 = !{!"_ZTSN4llvm3omp19OMPTgtExecModeFlagsE", !58, i64 0}
!61 = !{!"int", !58, i64 0}
!62 = !{!57, !58, i64 0}
!63 = !{!61, !61, i64 0}
!64 = !{!58, !58, i64 0}
!65 = !{!66, !61, i64 16}
!66 = !{!"_ZTSN4ompx5state11TeamStateTyE", !67, i64 0, !61, i64 28, !61, i64 32, !68, i64 40}
!67 = !{!"_ZTSN4ompx5state10ICVStateTyE", !61, i64 0, !61, i64 4, !61, i64 8, !61, i64 12, !61, i64 16, !61, i64 20, !61, i64 24}
!68 = !{!"p1 void", !69, i64 0}
!69 = !{!"any pointer", !58, i64 0}
!70 = !{!66, !61, i64 20}
!71 = !{!66, !61, i64 24}
!72 = !{!66, !61, i64 28}
!73 = !{!66, !61, i64 32}
!74 = !{!66, !68, i64 40}
!75 = !{!76, !76, i64 0}
!76 = !{!"p2 _ZTSN4ompx5state13ThreadStateTyE", !69, i64 0}
!77 = !{!78, !78, i64 0}
!78 = !{!"p1 _ZTS19KernelEnvironmentTy", !69, i64 0}
!79 = !{!80, !80, i64 0}
!80 = !{!"p1 _ZTS25KernelLaunchEnvironmentTy", !69, i64 0}
!81 = !{!82, !82, i64 0}
!82 = !{!"p2 _ZTS22DynamicScheduleTracker", !69, i64 0}
!83 = !{i32 1, i32 1025}
!84 = !{!85, !61, i64 0}
!85 = !{!"_ZTS19DeviceEnvironmentTy", !61, i64 0, !61, i64 4, !61, i64 8, !61, i64 12, !86, i64 16, !86, i64 24, !86, i64 32, !86, i64 40}
!86 = !{!"long", !58, i64 0}
!87 = !{!67, !61, i64 0}
!88 = !{!67, !61, i64 4}
!89 = !{!67, !61, i64 8}
!90 = !{!67, !61, i64 16}
!91 = !{!67, !61, i64 20}
!92 = !{!67, !61, i64 24}
!93 = !{i32 0, i32 1024}
!94 = !{!68, !68, i64 0}
!95 = distinct !{!95, !96}
!96 = !{!"llvm.loop.mustprogress"}
!97 = !{!98, !98, i64 0}
!98 = !{!"p1 _ZTSN4ompx5state13ThreadStateTyE", !69, i64 0}
!99 = !{!"branch_weights", !"expected", i32 2000, i32 1}
!100 = !{!101, !98, i64 32}
!101 = !{!"_ZTSN4ompx5state13ThreadStateTyE", !67, i64 0, !98, i64 32}
!102 = !{!103, !58, i64 0}
!103 = !{!"_ZTS19KernelEnvironmentTy", !57, i64 0, !104, i64 32, !105, i64 40}
!104 = !{!"p1 _ZTS7IdentTy", !69, i64 0}
!105 = !{!"p1 _ZTS20DynamicEnvironmentTy", !69, i64 0}
