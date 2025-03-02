; See ./README.md for how to maintain the LLVM IR in this test.

; REQUIRES: nvptx-registered-target

; RUN: opt -pass-remarks=kernel-info -passes=kernel-info \
; RUN:     -disable-output %s 2>&1 | \
; RUN:   FileCheck -match-full-lines %s

;  CHECK-NOT: remark:
;      CHECK: remark: test.c:0:0: in artificial function '[[OFF_FUNC:__omp_offloading_[a-f0-9_]*_h_l12]]_debug__', artificial alloca ('%[[#]]') for 'dyn_ptr' with static size of 8 bytes
; CHECK-NEXT: remark: test.c:14:9: in artificial function '[[OFF_FUNC]]_debug__', alloca ('%[[#]]') for 'i' with static size of 4 bytes
; CHECK-NEXT: remark: test.c:15:9: in artificial function '[[OFF_FUNC]]_debug__', alloca ('%[[#]]') for 'a' with static size of 8 bytes
; CHECK-NEXT: remark: <unknown>:0:0: in artificial function '[[OFF_FUNC]]_debug__', 'store' instruction accesses memory in flat address space
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

; CHECK-NEXT: remark: test.c:0:0: in artificial function '[[OFF_FUNC]]', artificial alloca ('%[[#]]') for 'dyn_ptr' with static size of 8 bytes
; CHECK-NEXT: remark: <unknown>:0:0: in artificial function '[[OFF_FUNC]]', 'store' instruction accesses memory in flat address space
; CHECK-NEXT: remark: test.c:12:1: in artificial function '[[OFF_FUNC]]', 'load' instruction ('%[[#]]') accesses memory in flat address space
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
;  CHECK-NOT: remark: {{.*: in function 'g',.*}}

; A lot of internal functions (e.g., __kmpc_target_init) come next, but we don't
; want to maintain a list of their allocas, calls, etc. in this test.

; ModuleID = 'test-openmp-nvptx64-nvidia-cuda-sm_70.bc'
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

@__omp_rtl_assume_teams_oversubscription = weak_odr hidden constant i32 0
@__omp_rtl_assume_threads_oversubscription = weak_odr hidden constant i32 0
@0 = private unnamed_addr constant [58 x i8] c";test.c;__omp_offloading_fd02_1116d6_h_l12_debug__;13;3;;\00", align 1
@1 = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 57, ptr @0 }, align 8
@__omp_offloading_fd02_1116d6_h_l12_dynamic_environment = weak_odr protected global %struct.DynamicEnvironmentTy zeroinitializer
@__omp_offloading_fd02_1116d6_h_l12_kernel_environment = weak_odr protected constant %struct.KernelEnvironmentTy { %struct.ConfigurationEnvironmentTy { i8 1, i8 1, i8 1, i32 1, i32 128, i32 -1, i32 -1, i32 0, i32 0 }, ptr @1, ptr @__omp_offloading_fd02_1116d6_h_l12_dynamic_environment }
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
@.str1027 = private unnamed_addr constant [48 x i8] c"/tmp/llvm/offload/DeviceRTL/src/Parallelism.cpp\00", align 1
@.str12 = private unnamed_addr constant [23 x i8] c"!mapping::isSPMDMode()\00", align 1
@__PRETTY_FUNCTION__.__kmpc_kernel_end_parallel = private unnamed_addr constant [34 x i8] c"void __kmpc_kernel_end_parallel()\00", align 1
@_ZL20KernelEnvironmentPtr = internal unnamed_addr addrspace(3) global ptr undef, align 8
@_ZL26KernelLaunchEnvironmentPtr = internal unnamed_addr addrspace(3) global ptr undef, align 8
@_ZN12_GLOBAL__N_122SharedMemorySmartStackE = internal addrspace(3) global %"struct.(anonymous namespace)::SharedMemorySmartStackTy" undef, align 16
@.str444 = private unnamed_addr constant [42 x i8] c"/tmp/llvm/offload/DeviceRTL/src/State.cpp\00", align 1
@.str747 = private unnamed_addr constant [33 x i8] c"NThreadsVar == Other.NThreadsVar\00", align 1
@__PRETTY_FUNCTION__._ZNK4ompx5state10ICVStateTy11assertEqualERKS1_ = private unnamed_addr constant [68 x i8] c"void ompx::state::ICVStateTy::assertEqual(const ICVStateTy &) const\00", align 1
@.str848 = private unnamed_addr constant [27 x i8] c"LevelVar == Other.LevelVar\00", align 1
@.str949 = private unnamed_addr constant [39 x i8] c"ActiveLevelVar == Other.ActiveLevelVar\00", align 1
@.str1050 = private unnamed_addr constant [47 x i8] c"MaxActiveLevelsVar == Other.MaxActiveLevelsVar\00", align 1
@.str1151 = private unnamed_addr constant [33 x i8] c"RunSchedVar == Other.RunSchedVar\00", align 1
@.str1252 = private unnamed_addr constant [43 x i8] c"RunSchedChunkVar == Other.RunSchedChunkVar\00", align 1
@.str13 = private unnamed_addr constant [43 x i8] c"ParallelTeamSize == Other.ParallelTeamSize\00", align 1
@__PRETTY_FUNCTION__._ZNK4ompx5state11TeamStateTy11assertEqualERS1_ = private unnamed_addr constant [64 x i8] c"void ompx::state::TeamStateTy::assertEqual(TeamStateTy &) const\00", align 1
@.str14 = private unnamed_addr constant [39 x i8] c"HasThreadState == Other.HasThreadState\00", align 1
@.str23 = private unnamed_addr constant [32 x i8] c"mapping::isSPMDMode() == IsSPMD\00", align 1
@__PRETTY_FUNCTION__._ZN4ompx5state18assumeInitialStateEb = private unnamed_addr constant [43 x i8] c"void ompx::state::assumeInitialState(bool)\00", align 1
@_ZL9ThreadDST = internal unnamed_addr addrspace(3) global ptr undef, align 8
@_ZN4ompx5state9TeamStateE = internal local_unnamed_addr addrspace(3) global %"struct.ompx::state::TeamStateTy" undef, align 8
@_ZN4ompx5state12ThreadStatesE = internal addrspace(3) global ptr undef, align 8

; Function Attrs: convergent noinline norecurse nounwind optnone
define internal void @__omp_offloading_fd02_1116d6_h_l12_debug__(ptr noalias noundef %0) #0 !dbg !18 {
  %2 = alloca ptr, align 8
  %3 = alloca i32, align 4
  %4 = alloca [2 x i32], align 4
  store ptr %0, ptr %2, align 8
    #dbg_declare(ptr %2, !25, !DIExpression(), !26)
  %5 = call i32 @__kmpc_target_init(ptr @__omp_offloading_fd02_1116d6_h_l12_kernel_environment, ptr %0), !dbg !27
  %6 = icmp eq i32 %5, -1, !dbg !27
  br i1 %6, label %7, label %8, !dbg !27

7:                                                ; preds = %1
    #dbg_declare(ptr %3, !28, !DIExpression(), !31)
    #dbg_declare(ptr %4, !32, !DIExpression(), !36)
  call void @f() #19, !dbg !37
  call void @g() #19, !dbg !38
  call void @__kmpc_target_deinit(), !dbg !39
  ret void, !dbg !40

8:                                                ; preds = %1
  ret void, !dbg !27
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define weak_odr protected ptx_kernel void @__omp_offloading_fd02_1116d6_h_l12(ptr noalias noundef %0) #1 !dbg !41 {
  %2 = alloca ptr, align 8
  store ptr %0, ptr %2, align 8
    #dbg_declare(ptr %2, !42, !DIExpression(), !43)
  %3 = load ptr, ptr %2, align 8, !dbg !44
  call void @__omp_offloading_fd02_1116d6_h_l12_debug__(ptr %3) #20, !dbg !44
  ret void, !dbg !44
}

; Function Attrs: convergent
declare void @f(...) #2

; Function Attrs: convergent noinline nounwind optnone
define hidden void @g() #3 !dbg !45 {
  %1 = alloca i32, align 4
  %2 = alloca [2 x i32], align 4
    #dbg_declare(ptr %1, !48, !DIExpression(), !49)
    #dbg_declare(ptr %2, !50, !DIExpression(), !51)
  call void @f() #19, !dbg !52
  call void @g() #19, !dbg !53
  ret void, !dbg !54
}

; Function Attrs: convergent mustprogress nounwind
define internal noundef range(i32 -1, 1024) i32 @__kmpc_target_init(ptr nofree noundef nonnull align 8 dereferenceable(48) %0, ptr nofree noundef nonnull align 8 dereferenceable(16) %1) #4 {
  %3 = alloca ptr, align 8
  %4 = getelementptr inbounds nuw i8, ptr %0, i64 2
  %5 = load i8, ptr %4, align 2, !tbaa !55
  %6 = and i8 %5, 2
  %7 = icmp eq i8 %6, 0
  %8 = load i8, ptr %0, align 8, !tbaa !61
  %9 = icmp ne i8 %8, 0
  br i1 %7, label %21, label %10

10:                                               ; preds = %2
  %11 = tail call range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %12 = icmp eq i32 %11, 0
  br i1 %12, label %13, label %14

13:                                               ; preds = %10
  store i32 1, ptr addrspace(3) @IsSPMDMode, align 4, !tbaa !62
  store i8 0, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN12_GLOBAL__N_122SharedMemorySmartStackE to ptr), i64 512) to ptr addrspace(3)), align 1, !tbaa !63
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(48) addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i8 noundef 0, i64 noundef 16, i1 noundef false)
  store i32 1, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 16) to ptr addrspace(3)), align 8, !tbaa !64
  store i32 1, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 20) to ptr addrspace(3)), align 4, !tbaa !69
  store i32 1, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 24) to ptr addrspace(3)), align 8, !tbaa !70
  store i32 1, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 28) to ptr addrspace(3)), align 4, !tbaa !71
  store i32 0, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 32) to ptr addrspace(3)), align 8, !tbaa !72
  store ptr null, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 40) to ptr addrspace(3)), align 8, !tbaa !73
  store ptr null, ptr addrspace(3) @_ZN4ompx5state12ThreadStatesE, align 8, !tbaa !74
  store ptr %0, ptr addrspace(3) @_ZL20KernelEnvironmentPtr, align 8, !tbaa !76
  store ptr %1, ptr addrspace(3) @_ZL26KernelLaunchEnvironmentPtr, align 8, !tbaa !78
  br label %18

14:                                               ; preds = %10
  %15 = zext nneg i32 %11 to i64
  %16 = getelementptr inbounds nuw [1024 x i8], ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN12_GLOBAL__N_122SharedMemorySmartStackE to ptr), i64 512), i64 0, i64 %15
  %17 = addrspacecast ptr %16 to ptr addrspace(3)
  store i8 0, ptr addrspace(3) %17, align 1, !tbaa !63
  br label %18

18:                                               ; preds = %14, %13
  br i1 %12, label %19, label %20

19:                                               ; preds = %18
  store ptr null, ptr addrspace(3) @_ZL9ThreadDST, align 8, !tbaa !80
  br label %20

20:                                               ; preds = %18, %19
  tail call void @_ZN4ompx11synchronize14threadsAlignedENS_6atomic10OrderingTyE(i32 poison) #21
  br label %37

21:                                               ; preds = %2
  %22 = tail call range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x(), !range !82
  %23 = add nsw i32 %22, -1
  %24 = and i32 %23, -32
  %25 = tail call range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x()
  %26 = icmp eq i32 %25, %24
  br i1 %26, label %27, label %31

27:                                               ; preds = %21
  store i32 0, ptr addrspace(3) @IsSPMDMode, align 4, !tbaa !62
  %28 = zext nneg i32 %25 to i64
  %29 = getelementptr inbounds nuw [1024 x i8], ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN12_GLOBAL__N_122SharedMemorySmartStackE to ptr), i64 512), i64 0, i64 %28
  %30 = addrspacecast ptr %29 to ptr addrspace(3)
  store i8 0, ptr addrspace(3) %30, align 1, !tbaa !63
  tail call void @llvm.memset.p0.i64(ptr noundef nonnull align 8 dereferenceable(48) addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i8 noundef 0, i64 noundef 16, i1 noundef false)
  store i32 1, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 16) to ptr addrspace(3)), align 8, !tbaa !64
  store i32 1, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 20) to ptr addrspace(3)), align 4, !tbaa !69
  store i32 1, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 24) to ptr addrspace(3)), align 8, !tbaa !70
  store i32 1, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 28) to ptr addrspace(3)), align 4, !tbaa !71
  store i32 0, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 32) to ptr addrspace(3)), align 8, !tbaa !72
  store ptr null, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 40) to ptr addrspace(3)), align 8, !tbaa !73
  store ptr null, ptr addrspace(3) @_ZN4ompx5state12ThreadStatesE, align 8, !tbaa !74
  store ptr %0, ptr addrspace(3) @_ZL20KernelEnvironmentPtr, align 8, !tbaa !76
  store ptr %1, ptr addrspace(3) @_ZL26KernelLaunchEnvironmentPtr, align 8, !tbaa !78
  br label %35

31:                                               ; preds = %21
  %32 = zext nneg i32 %25 to i64
  %33 = getelementptr inbounds nuw [1024 x i8], ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN12_GLOBAL__N_122SharedMemorySmartStackE to ptr), i64 512), i64 0, i64 %32
  %34 = addrspacecast ptr %33 to ptr addrspace(3)
  store i8 0, ptr addrspace(3) %34, align 1, !tbaa !63
  br label %35

35:                                               ; preds = %31, %27
  br i1 %26, label %36, label %37

36:                                               ; preds = %35
  store ptr null, ptr addrspace(3) @_ZL9ThreadDST, align 8, !tbaa !80
  br label %37

37:                                               ; preds = %36, %35, %20
  br i1 %7, label %100, label %38

38:                                               ; preds = %37
  %39 = load i32, ptr @__omp_rtl_debug_kind, align 4, !tbaa !62
  %40 = load i32, ptr addrspace(4) @__omp_rtl_device_environment, align 8, !tbaa !83
  %41 = and i32 %39, 1
  %42 = and i32 %41, %40
  %43 = icmp ne i32 %42, 0
  %44 = load i32, ptr addrspace(3) @_ZN4ompx5state9TeamStateE, align 8, !tbaa !86
  %45 = icmp ne i32 %44, 0
  %46 = select i1 %43, i1 %45, i1 false
  br i1 %46, label %47, label %48

47:                                               ; preds = %38
  tail call fastcc void @__assert_fail_internal(ptr noundef nonnull dereferenceable(33) @.str747, ptr noundef null, ptr noundef nonnull dereferenceable(66) @.str444, i32 noundef 193, ptr noundef nonnull dereferenceable(68) @__PRETTY_FUNCTION__._ZNK4ompx5state10ICVStateTy11assertEqualERKS1_) #22
  unreachable

48:                                               ; preds = %38
  %49 = icmp eq i32 %44, 0
  tail call void @llvm.assume(i1 noundef %49) #23
  %50 = load i32, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 4) to ptr addrspace(3)), align 4, !tbaa !87
  br i1 %43, label %51, label %54

51:                                               ; preds = %48
  %52 = icmp eq i32 %50, 0
  br i1 %52, label %54, label %53

53:                                               ; preds = %51
  tail call fastcc void @__assert_fail_internal(ptr noundef nonnull dereferenceable(27) @.str848, ptr noundef null, ptr noundef nonnull dereferenceable(66) @.str444, i32 noundef 194, ptr noundef nonnull dereferenceable(68) @__PRETTY_FUNCTION__._ZNK4ompx5state10ICVStateTy11assertEqualERKS1_) #22
  unreachable

54:                                               ; preds = %51, %48
  %55 = phi i32 [ 0, %51 ], [ %50, %48 ]
  %56 = icmp eq i32 %55, 0
  tail call void @llvm.assume(i1 noundef %56) #23
  %57 = load i32, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 8) to ptr addrspace(3)), align 8, !tbaa !88
  br i1 %43, label %58, label %61

58:                                               ; preds = %54
  %59 = icmp eq i32 %57, 0
  br i1 %59, label %61, label %60

60:                                               ; preds = %58
  tail call fastcc void @__assert_fail_internal(ptr noundef nonnull dereferenceable(39) @.str949, ptr noundef null, ptr noundef nonnull dereferenceable(66) @.str444, i32 noundef 195, ptr noundef nonnull dereferenceable(68) @__PRETTY_FUNCTION__._ZNK4ompx5state10ICVStateTy11assertEqualERKS1_) #22
  unreachable

61:                                               ; preds = %58, %54
  %62 = phi i32 [ 0, %58 ], [ %57, %54 ]
  %63 = icmp eq i32 %62, 0
  tail call void @llvm.assume(i1 noundef %63) #23
  %64 = load i32, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 16) to ptr addrspace(3)), align 8, !tbaa !89
  br i1 %43, label %65, label %68

65:                                               ; preds = %61
  %66 = icmp eq i32 %64, 1
  br i1 %66, label %68, label %67

67:                                               ; preds = %65
  tail call fastcc void @__assert_fail_internal(ptr noundef nonnull dereferenceable(47) @.str1050, ptr noundef null, ptr noundef nonnull dereferenceable(66) @.str444, i32 noundef 196, ptr noundef nonnull dereferenceable(68) @__PRETTY_FUNCTION__._ZNK4ompx5state10ICVStateTy11assertEqualERKS1_) #22
  unreachable

68:                                               ; preds = %65, %61
  %69 = phi i32 [ 1, %65 ], [ %64, %61 ]
  %70 = icmp eq i32 %69, 1
  tail call void @llvm.assume(i1 noundef %70) #23
  %71 = load i32, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 20) to ptr addrspace(3)), align 4, !tbaa !90
  br i1 %43, label %72, label %93

72:                                               ; preds = %68
  %73 = icmp eq i32 %71, 1
  br i1 %73, label %75, label %74

74:                                               ; preds = %72
  tail call fastcc void @__assert_fail_internal(ptr noundef nonnull dereferenceable(33) @.str1151, ptr noundef null, ptr noundef nonnull dereferenceable(66) @.str444, i32 noundef 197, ptr noundef nonnull dereferenceable(68) @__PRETTY_FUNCTION__._ZNK4ompx5state10ICVStateTy11assertEqualERKS1_) #22
  unreachable

75:                                               ; preds = %72
  %76 = icmp eq i32 1, 1
  tail call void @llvm.assume(i1 noundef %76) #23
  br i1 %43, label %77, label %95

77:                                               ; preds = %75
  %78 = load i32, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 24) to ptr addrspace(3)), align 8, !tbaa !91
  %79 = icmp eq i32 %78, 1
  br i1 %79, label %81, label %80

80:                                               ; preds = %77
  tail call fastcc void @__assert_fail_internal(ptr noundef nonnull dereferenceable(43) @.str1252, ptr noundef null, ptr noundef nonnull dereferenceable(66) @.str444, i32 noundef 198, ptr noundef nonnull dereferenceable(68) @__PRETTY_FUNCTION__._ZNK4ompx5state10ICVStateTy11assertEqualERKS1_) #22
  unreachable

81:                                               ; preds = %77
  %82 = load i32, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 28) to ptr addrspace(3)), align 4, !tbaa !71
  %83 = icmp eq i32 %82, 1
  br i1 %83, label %85, label %84

84:                                               ; preds = %81
  tail call fastcc void @__assert_fail_internal(ptr noundef nonnull dereferenceable(43) @.str13, ptr noundef null, ptr noundef nonnull dereferenceable(66) @.str444, i32 noundef 222, ptr noundef nonnull dereferenceable(64) @__PRETTY_FUNCTION__._ZNK4ompx5state11TeamStateTy11assertEqualERS1_) #22
  unreachable

85:                                               ; preds = %81
  %86 = load i32, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 32) to ptr addrspace(3)), align 8, !tbaa !72
  %87 = icmp eq i32 %86, 0
  br i1 %87, label %89, label %88

88:                                               ; preds = %85
  tail call fastcc void @__assert_fail_internal(ptr noundef nonnull dereferenceable(39) @.str14, ptr noundef null, ptr noundef nonnull dereferenceable(66) @.str444, i32 noundef 223, ptr noundef nonnull dereferenceable(64) @__PRETTY_FUNCTION__._ZNK4ompx5state11TeamStateTy11assertEqualERS1_) #22
  unreachable

89:                                               ; preds = %85
  %90 = load i32, ptr addrspace(3) @IsSPMDMode, align 4, !tbaa !62
  %91 = icmp eq i32 %90, 0
  br i1 %91, label %92, label %98

92:                                               ; preds = %89
  tail call fastcc void @__assert_fail_internal(ptr noundef nonnull dereferenceable(32) @.str23, ptr noundef null, ptr noundef nonnull dereferenceable(66) @.str444, i32 noundef 326, ptr noundef nonnull dereferenceable(43) @__PRETTY_FUNCTION__._ZN4ompx5state18assumeInitialStateEb) #22
  unreachable

93:                                               ; preds = %68
  %94 = icmp eq i32 %71, 1
  tail call void @llvm.assume(i1 noundef %94) #23
  br label %95

95:                                               ; preds = %75, %93
  %96 = load i32, ptr addrspace(3) @IsSPMDMode, align 4, !tbaa !62
  %97 = icmp ne i32 %96, 0
  br label %98

98:                                               ; preds = %89, %95
  %99 = phi i1 [ %97, %95 ], [ true, %89 ]
  tail call void @llvm.assume(i1 noundef %99) #23
  tail call void @_ZN4ompx11synchronize14threadsAlignedENS_6atomic10OrderingTyE(i32 poison) #21
  br label %130

100:                                              ; preds = %37
  %101 = tail call range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x(), !range !82
  %102 = add nsw i32 %101, -1
  %103 = and i32 %102, -32
  %104 = tail call range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !92
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
  call void @llvm.lifetime.start.p0(i64 noundef 8, ptr noundef nonnull align 8 dereferenceable(8) %3) #20
  tail call void @llvm.nvvm.barrier.sync(i32 noundef 8)
  %117 = call zeroext i1 @__kmpc_kernel_parallel(ptr noalias nocapture nofree noundef nonnull writeonly align 8 dereferenceable(8) %3) #20
  %118 = load ptr, ptr %3, align 8, !tbaa !93
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
  tail call fastcc void @__assert_fail_internal(ptr noundef nonnull dereferenceable(23) @.str12, ptr noundef null, ptr noundef nonnull dereferenceable(67) @.str15, i32 noundef 60, ptr noundef nonnull dereferenceable(36) @__PRETTY_FUNCTION__._ZL19genericStateMachineP7IdentTy) #22
  unreachable

126:                                              ; preds = %121
  %127 = icmp eq i32 %122, 0
  tail call void @llvm.assume(i1 noundef %127) #23
  tail call void %118(i32 noundef 0, i32 noundef %104) #24
  tail call void @__kmpc_kernel_end_parallel() #24
  br label %128

128:                                              ; preds = %126, %120
  tail call void @llvm.nvvm.barrier.sync(i32 noundef 8)
  call void @llvm.lifetime.end.p0(i64 noundef 8, ptr noundef nonnull %3) #20
  br label %116, !llvm.loop !94

129:                                              ; preds = %116
  call void @llvm.lifetime.end.p0(i64 noundef 8, ptr noundef nonnull %3) #20
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
define internal fastcc void @__assert_fail_internal(ptr noundef nonnull dereferenceable(8) %0, ptr noundef %1, ptr noundef nonnull dereferenceable(66) %2, i32 noundef range(i32 60, 905) %3, ptr noundef nonnull dereferenceable(20) %4) unnamed_addr #8 {
  %6 = icmp eq ptr %1, null
  br i1 %6, label %9, label %7

7:                                                ; preds = %5
  %8 = tail call noundef i32 (ptr, ...) @_ZN4ompx6printfEPKcz(ptr noundef nonnull dereferenceable(40) @.str, ptr noundef nonnull dereferenceable(66) %2, i32 noundef %3, ptr noundef nonnull dereferenceable(20) %4, ptr noundef nonnull %1, ptr noundef nonnull dereferenceable(8) %0) #24
  br label %11

9:                                                ; preds = %5
  %10 = tail call noundef i32 (ptr, ...) @_ZN4ompx6printfEPKcz(ptr noundef nonnull dereferenceable(35) @.str1, ptr noundef nonnull dereferenceable(66) %2, i32 noundef %3, ptr noundef nonnull dereferenceable(20) %4, ptr noundef nonnull dereferenceable(8) %0) #24
  br label %11

11:                                               ; preds = %9, %7
  tail call void @llvm.trap() #26
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
  %2 = load ptr, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 40) to ptr addrspace(3)), align 8, !tbaa !93
  store ptr %2, ptr %0, align 8, !tbaa !93
  %3 = icmp eq ptr %2, null
  br i1 %3, label %15, label %4

4:                                                ; preds = %1
  %5 = tail call noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #27, !range !92
  %6 = load i32, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 28) to ptr addrspace(3)), align 4, !tbaa !62
  %7 = icmp eq i32 %6, 0
  %8 = tail call range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x(), !range !82
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
  %1 = load i32, ptr @__omp_rtl_debug_kind, align 4, !tbaa !62
  %2 = load i32, ptr addrspace(4) @__omp_rtl_device_environment, align 8, !tbaa !83
  %3 = and i32 %1, 1
  %4 = and i32 %3, %2
  %5 = icmp ne i32 %4, 0
  %6 = load i32, ptr addrspace(3) @IsSPMDMode, align 4
  %7 = icmp ne i32 %6, 0
  %8 = select i1 %5, i1 %7, i1 false
  br i1 %8, label %9, label %10

9:                                                ; preds = %0
  tail call fastcc void @__assert_fail_internal(ptr noundef nonnull dereferenceable(23) @.str12, ptr noundef null, ptr noundef nonnull dereferenceable(72) @.str1027, i32 noundef 299, ptr noundef nonnull dereferenceable(34) @__PRETTY_FUNCTION__.__kmpc_kernel_end_parallel) #22
  unreachable

10:                                               ; preds = %0
  %11 = icmp eq i32 %6, 0
  tail call void @llvm.assume(i1 noundef %11) #23
  %12 = load i32, ptr @__omp_rtl_assume_no_thread_state, align 4, !tbaa !62
  %13 = icmp eq i32 %12, 0
  %14 = load i32, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 32) to ptr addrspace(3)), align 8
  %15 = icmp ne i32 %14, 0
  %16 = select i1 %13, i1 %15, i1 false
  br i1 %16, label %17, label %30

17:                                               ; preds = %10
  %18 = tail call noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #27, !range !92
  %19 = load ptr, ptr addrspace(3) @_ZN4ompx5state12ThreadStatesE, align 8, !tbaa !74
  %20 = zext nneg i32 %18 to i64
  %21 = getelementptr inbounds nuw ptr, ptr %19, i64 %20
  %22 = load ptr, ptr %21, align 8, !tbaa !96
  %23 = icmp eq ptr %22, null
  br i1 %23, label %30, label %24, !prof !98

24:                                               ; preds = %17
  %25 = getelementptr inbounds nuw i8, ptr %22, i64 32
  %26 = load ptr, ptr %25, align 8, !tbaa !99
  tail call void @free(ptr noundef nonnull dereferenceable(40) %22) #28
  %27 = load ptr, ptr addrspace(3) @_ZN4ompx5state12ThreadStatesE, align 8, !tbaa !74
  %28 = getelementptr inbounds nuw ptr, ptr %27, i64 %20
  store ptr %26, ptr %28, align 8, !tbaa !96
  %29 = load i32, ptr addrspace(3) @IsSPMDMode, align 4
  br label %30

30:                                               ; preds = %10, %17, %24
  %31 = phi i32 [ 0, %10 ], [ 0, %17 ], [ %29, %24 ]
  %32 = icmp ne i32 %31, 0
  %33 = select i1 %5, i1 %32, i1 false
  br i1 %33, label %34, label %35

34:                                               ; preds = %30
  tail call fastcc void @__assert_fail_internal(ptr noundef nonnull dereferenceable(23) @.str12, ptr noundef null, ptr noundef nonnull dereferenceable(72) @.str1027, i32 noundef 302, ptr noundef nonnull dereferenceable(34) @__PRETTY_FUNCTION__.__kmpc_kernel_end_parallel) #22
  unreachable

35:                                               ; preds = %30
  %36 = icmp eq i32 %31, 0
  tail call void @llvm.assume(i1 noundef %36) #23
  ret void
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(argmem: readwrite)
declare void @llvm.lifetime.end.p0(i64 immarg, ptr nocapture) #10

; Function Attrs: convergent mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite)
declare extern_weak void @free(ptr allocptr nocapture noundef) local_unnamed_addr #14

; Function Attrs: convergent mustprogress nounwind
define internal noundef i32 @_ZN4ompx6printfEPKcz(ptr noundef %0, ...) local_unnamed_addr #15 {
  %2 = alloca ptr, align 8
  call void @llvm.lifetime.start.p0(i64 noundef 8, ptr noundef nonnull align 8 %2) #29
  call void @llvm.va_start.p0(ptr noundef nonnull align 8 %2) #27
  %3 = load ptr, ptr %2, align 8, !tbaa !101
  %4 = call i32 @vprintf(ptr noundef %0, ptr noundef %3) #24
  call void @llvm.lifetime.end.p0(i64 noundef 8, ptr noundef nonnull %2) #20
  ret i32 %4
}

; Function Attrs: cold noreturn nounwind memory(inaccessiblemem: write)
declare void @llvm.trap() #16

; Function Attrs: nocallback nofree nosync nounwind willreturn
declare void @llvm.va_start.p0(ptr) #17

; Function Attrs: convergent nounwind
declare i32 @vprintf(ptr noundef, ptr noundef) local_unnamed_addr #18

; Function Attrs: convergent nocallback nounwind
declare void @llvm.nvvm.barrier0() #11

; Function Attrs: convergent mustprogress nounwind
define internal void @__kmpc_target_deinit() #4 {
  %1 = alloca ptr, align 8
  %2 = load i32, ptr addrspace(3) @IsSPMDMode, align 4, !tbaa !62
  %3 = icmp eq i32 %2, 0
  br i1 %3, label %4, label %27

4:                                                ; preds = %0
  %5 = tail call range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x(), !range !82
  %6 = add nsw i32 %5, -1
  %7 = and i32 %6, -32
  %8 = tail call range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x(), !range !92
  %9 = icmp eq i32 %8, %7
  br i1 %9, label %10, label %11

10:                                               ; preds = %4
  store ptr null, ptr addrspace(3) addrspacecast (ptr getelementptr inbounds nuw (i8, ptr addrspacecast (ptr addrspace(3) @_ZN4ompx5state9TeamStateE to ptr), i64 40) to ptr addrspace(3)), align 8, !tbaa !93
  br label %27

11:                                               ; preds = %4
  %12 = load ptr, ptr addrspace(3) @_ZL20KernelEnvironmentPtr, align 8, !tbaa !76
  %13 = load i8, ptr %12, align 8, !tbaa !103
  %14 = icmp eq i8 %13, 0
  br i1 %14, label %15, label %27

15:                                               ; preds = %11
  call void @llvm.lifetime.start.p0(i64 noundef 8, ptr noundef nonnull align 8 dereferenceable(8) %1) #29
  %16 = call zeroext i1 @__kmpc_kernel_parallel(ptr noalias nocapture nofree noundef nonnull writeonly align 8 dereferenceable(8) %1) #20
  %17 = load i32, ptr @__omp_rtl_debug_kind, align 4, !tbaa !62
  %18 = load i32, ptr addrspace(4) @__omp_rtl_device_environment, align 8, !tbaa !83
  %19 = and i32 %17, 1
  %20 = and i32 %19, %18
  %21 = icmp eq i32 %20, 0
  %22 = load ptr, ptr %1, align 8
  %23 = icmp eq ptr %22, null
  %24 = select i1 %21, i1 true, i1 %23
  br i1 %24, label %26, label %25

25:                                               ; preds = %15
  tail call fastcc void @__assert_fail_internal(ptr noundef nonnull dereferenceable(18) @.str2, ptr noundef null, ptr noundef nonnull dereferenceable(67) @.str15, i32 noundef 152, ptr noundef nonnull dereferenceable(28) @__PRETTY_FUNCTION__.__kmpc_target_deinit) #22
  unreachable

26:                                               ; preds = %15
  tail call void @llvm.assume(i1 noundef %23) #23
  call void @llvm.lifetime.end.p0(i64 noundef 8, ptr noundef nonnull %1) #20
  br label %27

27:                                               ; preds = %26, %11, %10, %0
  ret void
}

attributes #0 = { convergent noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_70" "target-features"="+ptx83,+sm_70" }
attributes #1 = { convergent mustprogress noinline norecurse nounwind optnone "frame-pointer"="all" "kernel" "no-trapping-math"="true" "omp_target_thread_limit"="128" "stack-protector-buffer-size"="8" "target-cpu"="sm_70" "target-features"="+ptx83,+sm_70" }
attributes #2 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_70" "target-features"="+ptx83,+sm_70" }
attributes #3 = { convergent noinline nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_70" "target-features"="+ptx83,+sm_70" }
attributes #4 = { convergent mustprogress nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_70" "target-features"="+ptx63,+ptx83,+sm_70" }
attributes #5 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #6 = { nocallback nofree nounwind willreturn memory(argmem: write) }
attributes #7 = { convergent mustprogress noinline norecurse nounwind "frame-pointer"="all" "llvm.assume"="ompx_aligned_barrier" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_70" "target-features"="+ptx63,+ptx83,+sm_70" }
attributes #8 = { cold convergent mustprogress noreturn nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_70" "target-features"="+ptx63,+ptx83,+sm_70" }
attributes #9 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #10 = { nocallback nofree nosync nounwind willreturn memory(argmem: readwrite) }
attributes #11 = { convergent nocallback nounwind }
attributes #12 = { convergent mustprogress nofree noinline norecurse nosync nounwind willreturn memory(read, argmem: write, inaccessiblemem: none) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_70" "target-features"="+ptx63,+ptx83,+sm_70" }
attributes #13 = { convergent mustprogress noinline nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_70" "target-features"="+ptx63,+ptx83,+sm_70" }
attributes #14 = { convergent mustprogress nounwind willreturn allockind("free") memory(argmem: readwrite, inaccessiblemem: readwrite) "alloc-family"="malloc" "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_70" "target-features"="+ptx63,+ptx83,+sm_70" }
attributes #15 = { convergent mustprogress nounwind "frame-pointer"="all" "no-builtin-printf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_70" "target-features"="+ptx63,+ptx83,+sm_70" }
attributes #16 = { cold noreturn nounwind memory(inaccessiblemem: write) }
attributes #17 = { nocallback nofree nosync nounwind willreturn }
attributes #18 = { convergent nounwind "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="sm_70" "target-features"="+ptx63,+ptx83,+sm_70" }
attributes #19 = { convergent }
attributes #20 = { nounwind }
attributes #21 = { convergent nounwind "llvm.assume"="ompx_aligned_barrier" }
attributes #22 = { convergent noreturn nounwind }
attributes #23 = { memory(write) }
attributes #24 = { convergent nounwind }
attributes #25 = { "llvm.assume"="ompx_aligned_barrier" }
attributes #26 = { noreturn }
attributes #27 = { nofree willreturn }
attributes #28 = { convergent nounwind willreturn }
attributes #29 = { nofree nounwind willreturn }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10}
!llvm.dbg.cu = !{!11}
!nvvm.annotations = !{!13}
!omp_offload.info = !{!14}
!llvm.ident = !{!15, !16, !15, !15, !15, !15, !15, !15, !15, !15, !15, !15, !15, !15, !15, !15, !15}
!nvvmir.version = !{!17}

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
!11 = distinct !DICompileUnit(language: DW_LANG_C11, file: !12, producer: "clang version 20.0.0git (/tmp/llvm/clang b9447c03a9ef2eed55b685a33511df86f7f94e89)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!12 = !DIFile(filename: "test.c", directory: "/tmp")
!13 = !{ptr @__omp_offloading_fd02_1116d6_h_l12, !"maxntidx", i32 128}
!14 = !{i32 0, i32 64770, i32 1119958, !"h", i32 12, i32 0, i32 0}
!15 = !{!"clang version 20.0.0git (/tmp/llvm/clang b9447c03a9ef2eed55b685a33511df86f7f94e89)"}
!16 = !{!"clang version 3.8.0 (tags/RELEASE_380/final)"}
!17 = !{i32 2, i32 0}
!18 = distinct !DISubprogram(name: "__omp_offloading_fd02_1116d6_h_l12_debug__", scope: !12, file: !12, line: 13, type: !19, scopeLine: 13, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !11, retainedNodes: !24)
!19 = !DISubroutineType(types: !20)
!20 = !{null, !21}
!21 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !22)
!22 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !23)
!23 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!24 = !{}
!25 = !DILocalVariable(name: "dyn_ptr", arg: 1, scope: !18, type: !21, flags: DIFlagArtificial)
!26 = !DILocation(line: 0, scope: !18)
!27 = !DILocation(line: 13, column: 3, scope: !18)
!28 = !DILocalVariable(name: "i", scope: !29, file: !12, line: 14, type: !30)
!29 = distinct !DILexicalBlock(scope: !18, file: !12, line: 13, column: 3)
!30 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!31 = !DILocation(line: 14, column: 9, scope: !29)
!32 = !DILocalVariable(name: "a", scope: !29, file: !12, line: 15, type: !33)
!33 = !DICompositeType(tag: DW_TAG_array_type, baseType: !30, size: 64, elements: !34)
!34 = !{!35}
!35 = !DISubrange(count: 2)
!36 = !DILocation(line: 15, column: 9, scope: !29)
!37 = !DILocation(line: 16, column: 5, scope: !29)
!38 = !DILocation(line: 17, column: 5, scope: !29)
!39 = !DILocation(line: 18, column: 3, scope: !29)
!40 = !DILocation(line: 18, column: 3, scope: !18)
!41 = distinct !DISubprogram(name: "__omp_offloading_fd02_1116d6_h_l12", scope: !12, file: !12, line: 12, type: !19, scopeLine: 12, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !11, retainedNodes: !24)
!42 = !DILocalVariable(name: "dyn_ptr", arg: 1, scope: !41, type: !21, flags: DIFlagArtificial)
!43 = !DILocation(line: 0, scope: !41)
!44 = !DILocation(line: 12, column: 1, scope: !41)
!45 = distinct !DISubprogram(name: "g", scope: !12, file: !12, line: 3, type: !46, scopeLine: 3, spFlags: DISPFlagDefinition, unit: !11, retainedNodes: !24)
!46 = !DISubroutineType(types: !47)
!47 = !{null}
!48 = !DILocalVariable(name: "i", scope: !45, file: !12, line: 4, type: !30)
!49 = !DILocation(line: 4, column: 7, scope: !45)
!50 = !DILocalVariable(name: "a", scope: !45, file: !12, line: 5, type: !33)
!51 = !DILocation(line: 5, column: 7, scope: !45)
!52 = !DILocation(line: 6, column: 3, scope: !45)
!53 = !DILocation(line: 7, column: 3, scope: !45)
!54 = !DILocation(line: 8, column: 1, scope: !45)
!55 = !{!56, !59, i64 2}
!56 = !{!"_ZTS26ConfigurationEnvironmentTy", !57, i64 0, !57, i64 1, !59, i64 2, !60, i64 4, !60, i64 8, !60, i64 12, !60, i64 16, !60, i64 20, !60, i64 24}
!57 = !{!"omnipotent char", !58, i64 0}
!58 = !{!"Simple C++ TBAA"}
!59 = !{!"_ZTSN4llvm3omp19OMPTgtExecModeFlagsE", !57, i64 0}
!60 = !{!"int", !57, i64 0}
!61 = !{!56, !57, i64 0}
!62 = !{!60, !60, i64 0}
!63 = !{!57, !57, i64 0}
!64 = !{!65, !60, i64 16}
!65 = !{!"_ZTSN4ompx5state11TeamStateTyE", !66, i64 0, !60, i64 28, !60, i64 32, !67, i64 40}
!66 = !{!"_ZTSN4ompx5state10ICVStateTyE", !60, i64 0, !60, i64 4, !60, i64 8, !60, i64 12, !60, i64 16, !60, i64 20, !60, i64 24}
!67 = !{!"p1 void", !68, i64 0}
!68 = !{!"any pointer", !57, i64 0}
!69 = !{!65, !60, i64 20}
!70 = !{!65, !60, i64 24}
!71 = !{!65, !60, i64 28}
!72 = !{!65, !60, i64 32}
!73 = !{!65, !67, i64 40}
!74 = !{!75, !75, i64 0}
!75 = !{!"p2 _ZTSN4ompx5state13ThreadStateTyE", !68, i64 0}
!76 = !{!77, !77, i64 0}
!77 = !{!"p1 _ZTS19KernelEnvironmentTy", !68, i64 0}
!78 = !{!79, !79, i64 0}
!79 = !{!"p1 _ZTS25KernelLaunchEnvironmentTy", !68, i64 0}
!80 = !{!81, !81, i64 0}
!81 = !{!"p2 _ZTS22DynamicScheduleTracker", !68, i64 0}
!82 = !{i32 1, i32 1025}
!83 = !{!84, !60, i64 0}
!84 = !{!"_ZTS19DeviceEnvironmentTy", !60, i64 0, !60, i64 4, !60, i64 8, !60, i64 12, !85, i64 16, !85, i64 24, !85, i64 32, !85, i64 40}
!85 = !{!"long", !57, i64 0}
!86 = !{!66, !60, i64 0}
!87 = !{!66, !60, i64 4}
!88 = !{!66, !60, i64 8}
!89 = !{!66, !60, i64 16}
!90 = !{!66, !60, i64 20}
!91 = !{!66, !60, i64 24}
!92 = !{i32 0, i32 1024}
!93 = !{!67, !67, i64 0}
!94 = distinct !{!94, !95}
!95 = !{!"llvm.loop.mustprogress"}
!96 = !{!97, !97, i64 0}
!97 = !{!"p1 _ZTSN4ompx5state13ThreadStateTyE", !68, i64 0}
!98 = !{!"branch_weights", !"expected", i32 2000, i32 1}
!99 = !{!100, !97, i64 32}
!100 = !{!"_ZTSN4ompx5state13ThreadStateTyE", !66, i64 0, !97, i64 32}
!101 = !{!102, !102, i64 0}
!102 = !{!"p1 omnipotent char", !68, i64 0}
!103 = !{!104, !57, i64 0}
!104 = !{!"_ZTS19KernelEnvironmentTy", !56, i64 0, !105, i64 32, !106, i64 40}
!105 = !{!"p1 _ZTS7IdentTy", !68, i64 0}
!106 = !{!"p1 _ZTS20DynamicEnvironmentTy", !68, i64 0}
