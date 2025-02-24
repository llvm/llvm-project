; See ./README.md for how to maintain the LLVM IR in this test.

; REQUIRES: amdgpu-registered-target

; RUN: opt -pass-remarks=kernel-info -passes=kernel-info \
; RUN:     -disable-output %s 2>&1 | \
; RUN:   FileCheck -match-full-lines %s

;  CHECK-NOT: remark:
;      CHECK: remark: test.c:0:0: in artificial function '[[OFF_FUNC:__omp_offloading_[a-f0-9_]*_h_l12]]_debug__', artificial alloca ('%[[#]]') for 'dyn_ptr' with static size of 8 bytes
; CHECK-NEXT: remark: test.c:14:9: in artificial function '[[OFF_FUNC]]_debug__', alloca ('%[[#]]') for 'i' with static size of 4 bytes
; CHECK-NEXT: remark: test.c:15:9: in artificial function '[[OFF_FUNC]]_debug__', alloca ('%[[#]]') for 'a' with static size of 8 bytes
; CHECK-NEXT: remark: <unknown>:0:0: in artificial function '[[OFF_FUNC]]_debug__', 'store' instruction accesses memory in flat address space
; CHECK-NEXT: remark: test.c:13:3: in artificial function '[[OFF_FUNC]]_debug__', direct call, callee is '@__kmpc_target_init'
; CHECK-NEXT: remark: test.c:16:5: in artificial function '[[OFF_FUNC]]_debug__', direct call, callee is '@f'
; CHECK-NEXT: remark: test.c:17:5: in artificial function '[[OFF_FUNC]]_debug__', direct call to defined function, callee is 'g'
; CHECK-NEXT: remark: test.c:18:3: in artificial function '[[OFF_FUNC]]_debug__', direct call, callee is '@__kmpc_target_deinit'
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', ExternalNotKernel = 0
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', amdgpu-max-num-workgroups[0] = 4294967295
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', amdgpu-max-num-workgroups[1] = 4294967295
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', amdgpu-max-num-workgroups[2] = 4294967295
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', amdgpu-flat-work-group-size[0] = 1
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', amdgpu-flat-work-group-size[1] = 1024
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', amdgpu-waves-per-eu[0] = 4
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', amdgpu-waves-per-eu[1] = 10
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', Allocas = 3
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', AllocasStaticSizeSum = 20
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', AllocasDyn = 0
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', DirectCalls = 4
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', IndirectCalls = 0
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', DirectCallsToDefinedFunctions = 1
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', InlineAssemblyCalls = 0
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', Invokes = 0
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', FlatAddrspaceAccesses = 1

; CHECK-NEXT: remark: test.c:0:0: in artificial function '[[OFF_FUNC]]', artificial alloca ('%[[#]]') for 'dyn_ptr' with static size of 8 bytes
; CHECK-NEXT: remark: <unknown>:0:0: in artificial function '[[OFF_FUNC]]', 'store' instruction accesses memory in flat address space
; CHECK-NEXT: remark: test.c:12:1: in artificial function '[[OFF_FUNC]]', 'load' instruction ('%[[#]]') accesses memory in flat address space
; CHECK-NEXT: remark: test.c:12:1: in artificial function '[[OFF_FUNC]]', direct call to defined function, callee is artificial '[[OFF_FUNC]]_debug__'
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', ExternalNotKernel = 0
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', omp_target_thread_limit = 256
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', amdgpu-max-num-workgroups[0] = 4294967295
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', amdgpu-max-num-workgroups[1] = 4294967295
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', amdgpu-max-num-workgroups[2] = 4294967295
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', amdgpu-flat-work-group-size[0] = 1
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', amdgpu-flat-work-group-size[1] = 256
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', amdgpu-waves-per-eu[0] = 1
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', amdgpu-waves-per-eu[1] = 10
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
; CHECK-NEXT: remark: test.c:3:0: in function 'g', amdgpu-max-num-workgroups[0] = 4294967295
; CHECK-NEXT: remark: test.c:3:0: in function 'g', amdgpu-max-num-workgroups[1] = 4294967295
; CHECK-NEXT: remark: test.c:3:0: in function 'g', amdgpu-max-num-workgroups[2] = 4294967295
; CHECK-NEXT: remark: test.c:3:0: in function 'g', amdgpu-flat-work-group-size[0] = 1
; CHECK-NEXT: remark: test.c:3:0: in function 'g', amdgpu-flat-work-group-size[1] = 1024
; CHECK-NEXT: remark: test.c:3:0: in function 'g', amdgpu-waves-per-eu[0] = 4
; CHECK-NEXT: remark: test.c:3:0: in function 'g', amdgpu-waves-per-eu[1] = 10
; CHECK-NEXT: remark: test.c:3:0: in function 'g', Allocas = 2
; CHECK-NEXT: remark: test.c:3:0: in function 'g', AllocasStaticSizeSum = 12
; CHECK-NEXT: remark: test.c:3:0: in function 'g', AllocasDyn = 0
; CHECK-NEXT: remark: test.c:3:0: in function 'g', DirectCalls = 2
; CHECK-NEXT: remark: test.c:3:0: in function 'g', IndirectCalls = 0
; CHECK-NEXT: remark: test.c:3:0: in function 'g', DirectCallsToDefinedFunctions = 1
; CHECK-NEXT: remark: test.c:3:0: in function 'g', InlineAssemblyCalls = 0
; CHECK-NEXT: remark: test.c:3:0: in function 'g', Invokes = 0
; CHECK-NEXT: remark: test.c:3:0: in function 'g', FlatAddrspaceAccesses = 0
;  CHECK-NOT: {{.}}

; ModuleID = 'test-openmp-amdgcn-amd-amdhsa-gfx906.bc'
source_filename = "test.c"
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-p7:160:256:256:32-p8:128:128-p9:192:256:256:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7:8:9"
target triple = "amdgcn-amd-amdhsa"

%struct.ident_t = type { i32, i32, i32, i32, ptr }
%struct.DynamicEnvironmentTy = type { i16 }
%struct.KernelEnvironmentTy = type { %struct.ConfigurationEnvironmentTy, ptr, ptr }
%struct.ConfigurationEnvironmentTy = type { i8, i8, i8, i32, i32, i32, i32, i32, i32 }

@__omp_rtl_debug_kind = weak_odr hidden addrspace(1) constant i32 0
@__omp_rtl_assume_teams_oversubscription = weak_odr hidden addrspace(1) constant i32 0
@__omp_rtl_assume_threads_oversubscription = weak_odr hidden addrspace(1) constant i32 0
@__omp_rtl_assume_no_thread_state = weak_odr hidden addrspace(1) constant i32 0
@__omp_rtl_assume_no_nested_parallelism = weak_odr hidden addrspace(1) constant i32 0
@0 = private unnamed_addr constant [57 x i8] c";test.c;__omp_offloading_fd02_727e9_h_l12_debug__;13;3;;\00", align 1
@1 = private unnamed_addr addrspace(1) constant %struct.ident_t { i32 0, i32 2, i32 0, i32 56, ptr @0 }, align 8
@__omp_offloading_fd02_727e9_h_l12_dynamic_environment = weak_odr protected addrspace(1) global %struct.DynamicEnvironmentTy zeroinitializer
@__omp_offloading_fd02_727e9_h_l12_kernel_environment = weak_odr protected addrspace(1) constant %struct.KernelEnvironmentTy { %struct.ConfigurationEnvironmentTy { i8 1, i8 1, i8 1, i32 1, i32 256, i32 -1, i32 -1, i32 0, i32 0 }, ptr addrspacecast (ptr addrspace(1) @1 to ptr), ptr addrspacecast (ptr addrspace(1) @__omp_offloading_fd02_727e9_h_l12_dynamic_environment to ptr) }
@__oclc_ABI_version = weak_odr hidden local_unnamed_addr addrspace(4) constant i32 500

; Function Attrs: convergent noinline norecurse nounwind optnone
define internal void @__omp_offloading_fd02_727e9_h_l12_debug__(ptr noalias noundef %0) #0 !dbg !15 {
  %2 = alloca ptr, align 8, addrspace(5)
  %3 = alloca i32, align 4, addrspace(5)
  %4 = alloca [2 x i32], align 4, addrspace(5)
  %5 = addrspacecast ptr addrspace(5) %2 to ptr
  %6 = addrspacecast ptr addrspace(5) %3 to ptr
  %7 = addrspacecast ptr addrspace(5) %4 to ptr
  store ptr %0, ptr %5, align 8
    #dbg_declare(ptr addrspace(5) %2, !23, !DIExpression(), !24)
  %8 = call i32 @__kmpc_target_init(ptr addrspacecast (ptr addrspace(1) @__omp_offloading_fd02_727e9_h_l12_kernel_environment to ptr), ptr %0), !dbg !25
  %9 = icmp eq i32 %8, -1, !dbg !25
  br i1 %9, label %10, label %11, !dbg !25

10:                                               ; preds = %1
    #dbg_declare(ptr addrspace(5) %3, !26, !DIExpression(), !29)
    #dbg_declare(ptr addrspace(5) %4, !30, !DIExpression(), !34)
  call void @f() #4, !dbg !35
  call void @g() #4, !dbg !36
  call void @__kmpc_target_deinit(), !dbg !37
  ret void, !dbg !38

11:                                               ; preds = %1
  ret void, !dbg !25
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define weak_odr protected amdgpu_kernel void @__omp_offloading_fd02_727e9_h_l12(ptr noalias noundef %0) #1 !dbg !39 {
  %2 = alloca ptr, align 8, addrspace(5)
  %3 = addrspacecast ptr addrspace(5) %2 to ptr
  store ptr %0, ptr %3, align 8
    #dbg_declare(ptr addrspace(5) %2, !40, !DIExpression(), !41)
  %4 = load ptr, ptr %3, align 8, !dbg !42
  call void @__omp_offloading_fd02_727e9_h_l12_debug__(ptr %4) #5, !dbg !42
  ret void, !dbg !42
}

declare i32 @__kmpc_target_init(ptr, ptr)

; Function Attrs: convergent
declare void @f(...) #2

declare void @__kmpc_target_deinit()

; Function Attrs: convergent noinline nounwind optnone
define hidden void @g() #3 !dbg !43 {
  %1 = alloca i32, align 4, addrspace(5)
  %2 = alloca [2 x i32], align 4, addrspace(5)
  %3 = addrspacecast ptr addrspace(5) %1 to ptr
  %4 = addrspacecast ptr addrspace(5) %2 to ptr
    #dbg_declare(ptr addrspace(5) %1, !46, !DIExpression(), !47)
    #dbg_declare(ptr addrspace(5) %2, !48, !DIExpression(), !49)
  call void @f() #4, !dbg !50
  call void @g() #4, !dbg !51
  ret void, !dbg !52
}

attributes #0 = { convergent noinline norecurse nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx906" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" }
attributes #1 = { convergent mustprogress noinline norecurse nounwind optnone "amdgpu-flat-work-group-size"="1,256" "frame-pointer"="all" "kernel" "no-trapping-math"="true" "omp_target_thread_limit"="256" "stack-protector-buffer-size"="8" "target-cpu"="gfx906" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" "uniform-work-group-size"="true" }
attributes #2 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx906" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" }
attributes #3 = { convergent noinline nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx906" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" }
attributes #4 = { convergent }
attributes #5 = { nounwind }

!llvm.dbg.cu = !{!0}
!omp_offload.info = !{!2}
!llvm.module.flags = !{!3, !4, !5, !6, !7, !8, !9, !10, !11}
!llvm.ident = !{!12, !13, !13, !13, !13, !13, !13, !13, !13, !13, !13, !13, !13, !13, !13, !13, !13}
!opencl.ocl.version = !{!14, !14, !14, !14, !14, !14, !14, !14, !14, !14, !14, !14, !14, !14, !14, !14}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 20.0.0git (/tmp/llvm/clang b9447c03a9ef2eed55b685a33511df86f7f94e89)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp", checksumkind: CSK_MD5, checksum: "27a878d5e894ab6d41bfe96f997f8821")
!2 = !{i32 0, i32 64770, i32 468969, !"h", i32 12, i32 0, i32 0}
!3 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!4 = !{i32 7, !"Dwarf Version", i32 5}
!5 = !{i32 2, !"Debug Info Version", i32 3}
!6 = !{i32 1, !"wchar_size", i32 4}
!7 = !{i32 7, !"openmp", i32 51}
!8 = !{i32 7, !"openmp-device", i32 51}
!9 = !{i32 8, !"PIC Level", i32 2}
!10 = !{i32 7, !"frame-pointer", i32 2}
!11 = !{i32 4, !"amdgpu_hostcall", i32 1}
!12 = !{!"clang version 20.0.0git (/tmp/llvm/clang b9447c03a9ef2eed55b685a33511df86f7f94e89)"}
!13 = !{!"AMD clang version 17.0.0 (https://github.com/RadeonOpenCompute/llvm-project roc-6.0.2 24012 af27734ed982b52a9f1be0f035ac91726fc697e4)"}
!14 = !{i32 2, i32 0}
!15 = distinct !DISubprogram(name: "__omp_offloading_fd02_727e9_h_l12_debug__", scope: !16, file: !16, line: 13, type: !17, scopeLine: 13, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !22)
!16 = !DIFile(filename: "test.c", directory: "/tmp")
!17 = !DISubroutineType(types: !18)
!18 = !{null, !19}
!19 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !20)
!20 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !21)
!21 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!22 = !{}
!23 = !DILocalVariable(name: "dyn_ptr", arg: 1, scope: !15, type: !19, flags: DIFlagArtificial)
!24 = !DILocation(line: 0, scope: !15)
!25 = !DILocation(line: 13, column: 3, scope: !15)
!26 = !DILocalVariable(name: "i", scope: !27, file: !16, line: 14, type: !28)
!27 = distinct !DILexicalBlock(scope: !15, file: !16, line: 13, column: 3)
!28 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!29 = !DILocation(line: 14, column: 9, scope: !27)
!30 = !DILocalVariable(name: "a", scope: !27, file: !16, line: 15, type: !31)
!31 = !DICompositeType(tag: DW_TAG_array_type, baseType: !28, size: 64, elements: !32)
!32 = !{!33}
!33 = !DISubrange(count: 2)
!34 = !DILocation(line: 15, column: 9, scope: !27)
!35 = !DILocation(line: 16, column: 5, scope: !27)
!36 = !DILocation(line: 17, column: 5, scope: !27)
!37 = !DILocation(line: 18, column: 3, scope: !27)
!38 = !DILocation(line: 18, column: 3, scope: !15)
!39 = distinct !DISubprogram(name: "__omp_offloading_fd02_727e9_h_l12", scope: !16, file: !16, line: 12, type: !17, scopeLine: 12, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !22)
!40 = !DILocalVariable(name: "dyn_ptr", arg: 1, scope: !39, type: !19, flags: DIFlagArtificial)
!41 = !DILocation(line: 0, scope: !39)
!42 = !DILocation(line: 12, column: 1, scope: !39)
!43 = distinct !DISubprogram(name: "g", scope: !16, file: !16, line: 3, type: !44, scopeLine: 3, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !22)
!44 = !DISubroutineType(types: !45)
!45 = !{null}
!46 = !DILocalVariable(name: "i", scope: !43, file: !16, line: 4, type: !28)
!47 = !DILocation(line: 4, column: 7, scope: !43)
!48 = !DILocalVariable(name: "a", scope: !43, file: !16, line: 5, type: !31)
!49 = !DILocation(line: 5, column: 7, scope: !43)
!50 = !DILocation(line: 6, column: 3, scope: !43)
!51 = !DILocation(line: 7, column: 3, scope: !43)
!52 = !DILocation(line: 8, column: 1, scope: !43)
