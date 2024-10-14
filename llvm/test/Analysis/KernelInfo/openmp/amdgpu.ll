; See ./README.md for how to maintain the LLVM IR in this test.

; REQUIRES: amdgpu-registered-target

; RUN: opt -pass-remarks=kernel-info -passes=kernel-info \
; RUN:     -disable-output %s 2>&1 | \
; RUN:   FileCheck -match-full-lines %s

;  CHECK-NOT: remark:
;      CHECK: remark: test.c:0:0: in artificial function '[[OFF_FUNC:__omp_offloading_[a-f0-9_]*_h_l12]]_debug__', artificial alloca 'dyn_ptr' with static size of 8 bytes
; CHECK-NEXT: remark: test.c:14:9: in artificial function '[[OFF_FUNC]]_debug__', alloca 'i' with static size of 4 bytes
; CHECK-NEXT: remark: test.c:15:9: in artificial function '[[OFF_FUNC]]_debug__', alloca 'a' with static size of 8 bytes
; CHECK-NEXT: remark: <unknown>:0:0: in artificial function '[[OFF_FUNC]]_debug__', 'store' instruction accesses memory in flat address space
; CHECK-NEXT: remark: test.c:13:3: in artificial function '[[OFF_FUNC]]_debug__', direct call, callee is '@__kmpc_target_init'
; CHECK-NEXT: remark: test.c:16:5: in artificial function '[[OFF_FUNC]]_debug__', direct call, callee is '@f'
; CHECK-NEXT: remark: test.c:17:5: in artificial function '[[OFF_FUNC]]_debug__', direct call to defined function, callee is 'g'
; CHECK-NEXT: remark: test.c:18:3: in artificial function '[[OFF_FUNC]]_debug__', direct call, callee is '@__kmpc_target_deinit'
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', ExternalNotKernel = 0
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', omp_target_thread_limit = 256
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', amdgpu-max-num-workgroups[0] = 0
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', amdgpu-max-num-workgroups[1] = 0
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', amdgpu-max-num-workgroups[2] = 0
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', amdgpu-flat-work-group-size[0] = 1
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', amdgpu-flat-work-group-size[1] = 256
; CHECK-NEXT: remark: test.c:13:0: in artificial function '[[OFF_FUNC]]_debug__', amdgpu-waves-per-eu[0] = 1
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

; CHECK-NEXT: remark: test.c:0:0: in artificial function '[[OFF_FUNC]]', artificial alloca 'dyn_ptr' with static size of 8 bytes
; CHECK-NEXT: remark: <unknown>:0:0: in artificial function '[[OFF_FUNC]]', 'store' instruction accesses memory in flat address space
; CHECK-NEXT: remark: test.c:12:1: in artificial function '[[OFF_FUNC]]', 'load' instruction ('%[[#]]') accesses memory in flat address space
; CHECK-NEXT: remark: test.c:12:1: in artificial function '[[OFF_FUNC]]', direct call to defined function, callee is artificial '[[OFF_FUNC]]_debug__'
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', ExternalNotKernel = 0
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', amdgpu-max-num-workgroups[0] = 0
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', amdgpu-max-num-workgroups[1] = 0
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', amdgpu-max-num-workgroups[2] = 0
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', amdgpu-flat-work-group-size[0] = 1
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', amdgpu-flat-work-group-size[1] = 1024
; CHECK-NEXT: remark: test.c:12:0: in artificial function '[[OFF_FUNC]]', amdgpu-waves-per-eu[0] = 4
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

; CHECK-NEXT: remark: test.c:4:7: in function 'g', alloca 'i' with static size of 4 bytes
; CHECK-NEXT: remark: test.c:5:7: in function 'g', alloca 'a' with static size of 8 bytes
; CHECK-NEXT: remark: test.c:6:3: in function 'g', direct call, callee is '@f'
; CHECK-NEXT: remark: test.c:7:3: in function 'g', direct call to defined function, callee is 'g'
; CHECK-NEXT: remark: test.c:3:0: in function 'g', ExternalNotKernel = 1
; CHECK-NEXT: remark: test.c:3:0: in function 'g', amdgpu-max-num-workgroups[0] = 0
; CHECK-NEXT: remark: test.c:3:0: in function 'g', amdgpu-max-num-workgroups[1] = 0
; CHECK-NEXT: remark: test.c:3:0: in function 'g', amdgpu-max-num-workgroups[2] = 0
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


; ModuleID = 'test-openmp-amdgcn-amd-amdhsa.bc'
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
@0 = private unnamed_addr constant [57 x i8] c";test.c;__omp_offloading_fd02_71f35_h_l12_debug__;13;3;;\00", align 1
@1 = private unnamed_addr addrspace(1) constant %struct.ident_t { i32 0, i32 2, i32 0, i32 56, ptr @0 }, align 8
@__omp_offloading_fd02_71f35_h_l12_dynamic_environment = weak_odr protected addrspace(1) global %struct.DynamicEnvironmentTy zeroinitializer
@__omp_offloading_fd02_71f35_h_l12_kernel_environment = weak_odr protected addrspace(1) constant %struct.KernelEnvironmentTy { %struct.ConfigurationEnvironmentTy { i8 1, i8 1, i8 1, i32 1, i32 256, i32 -1, i32 -1, i32 0, i32 0 }, ptr addrspacecast (ptr addrspace(1) @1 to ptr), ptr addrspacecast (ptr addrspace(1) @__omp_offloading_fd02_71f35_h_l12_dynamic_environment to ptr) }
@__oclc_ABI_version = weak_odr hidden local_unnamed_addr addrspace(4) constant i32 500

; Function Attrs: convergent noinline norecurse nounwind optnone
define internal void @__omp_offloading_fd02_71f35_h_l12_debug__(ptr noalias noundef %dyn_ptr) #0 !dbg !16 {
entry:
  %dyn_ptr.addr = alloca ptr, align 8, addrspace(5)
  %i = alloca i32, align 4, addrspace(5)
  %a = alloca [2 x i32], align 4, addrspace(5)
  %dyn_ptr.addr.ascast = addrspacecast ptr addrspace(5) %dyn_ptr.addr to ptr
  %i.ascast = addrspacecast ptr addrspace(5) %i to ptr
  %a.ascast = addrspacecast ptr addrspace(5) %a to ptr
  store ptr %dyn_ptr, ptr %dyn_ptr.addr.ascast, align 8
  tail call void @llvm.dbg.declare(metadata ptr addrspace(5) %dyn_ptr.addr, metadata !24, metadata !DIExpression()), !dbg !25
  %0 = call i32 @__kmpc_target_init(ptr addrspacecast (ptr addrspace(1) @__omp_offloading_fd02_71f35_h_l12_kernel_environment to ptr), ptr %dyn_ptr), !dbg !26
  %exec_user_code = icmp eq i32 %0, -1, !dbg !26
  br i1 %exec_user_code, label %user_code.entry, label %worker.exit, !dbg !26

user_code.entry:                                  ; preds = %entry
  tail call void @llvm.dbg.declare(metadata ptr addrspace(5) %i, metadata !27, metadata !DIExpression()), !dbg !30
  tail call void @llvm.dbg.declare(metadata ptr addrspace(5) %a, metadata !31, metadata !DIExpression()), !dbg !35
  call void @f() #5, !dbg !36
  call void @g() #5, !dbg !37
  call void @__kmpc_target_deinit(), !dbg !38
  ret void, !dbg !39

worker.exit:                                      ; preds = %entry
  ret void, !dbg !26
}

declare i32 @__kmpc_target_init(ptr, ptr)

; Function Attrs: convergent
declare void @f(...) #1

declare void @__kmpc_target_deinit()

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define weak_odr protected amdgpu_kernel void @__omp_offloading_fd02_71f35_h_l12(ptr noalias noundef %dyn_ptr) #2 !dbg !40 {
entry:
  %dyn_ptr.addr = alloca ptr, align 8, addrspace(5)
  %dyn_ptr.addr.ascast = addrspacecast ptr addrspace(5) %dyn_ptr.addr to ptr
  store ptr %dyn_ptr, ptr %dyn_ptr.addr.ascast, align 8
  tail call void @llvm.dbg.declare(metadata ptr addrspace(5) %dyn_ptr.addr, metadata !41, metadata !DIExpression()), !dbg !42
  %0 = load ptr, ptr %dyn_ptr.addr.ascast, align 8, !dbg !43
  call void @__omp_offloading_fd02_71f35_h_l12_debug__(ptr %0) #6, !dbg !43
  ret void, !dbg !43
}

; Function Attrs: convergent noinline nounwind optnone
define hidden void @g() #3 !dbg !44 {
entry:
  %i = alloca i32, align 4, addrspace(5)
  %a = alloca [2 x i32], align 4, addrspace(5)
  %i.ascast = addrspacecast ptr addrspace(5) %i to ptr
  %a.ascast = addrspacecast ptr addrspace(5) %a to ptr
  tail call void @llvm.dbg.declare(metadata ptr addrspace(5) %i, metadata !47, metadata !DIExpression()), !dbg !48
  tail call void @llvm.dbg.declare(metadata ptr addrspace(5) %a, metadata !49, metadata !DIExpression()), !dbg !50
  call void @f() #5, !dbg !51
  call void @g() #5, !dbg !52
  ret void, !dbg !53
}

; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare void @llvm.dbg.declare(metadata, metadata, metadata) #4

attributes #0 = { convergent noinline norecurse nounwind optnone "amdgpu-flat-work-group-size"="1,256" "frame-pointer"="all" "no-trapping-math"="true" "omp_target_thread_limit"="256" "stack-protector-buffer-size"="8" "target-cpu"="gfx906" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" }
attributes #1 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx906" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" }
attributes #2 = { convergent mustprogress noinline norecurse nounwind optnone "frame-pointer"="all" "kernel" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx906" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" "uniform-work-group-size"="true" }
attributes #3 = { convergent noinline nounwind optnone "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="gfx906" "target-features"="+16-bit-insts,+ci-insts,+dl-insts,+dot1-insts,+dot10-insts,+dot2-insts,+dot7-insts,+dpp,+gfx8-insts,+gfx9-insts,+s-memrealtime,+s-memtime-inst,+wavefrontsize64" }
attributes #4 = { nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #5 = { convergent }
attributes #6 = { nounwind }

!llvm.dbg.cu = !{!0}
!omp_offload.info = !{!2}
!nvvm.annotations = !{!3}
!llvm.module.flags = !{!4, !5, !6, !7, !8, !9, !10, !11, !12}
!llvm.ident = !{!13, !14, !14, !14, !14, !14, !14, !14, !14, !14, !14, !14, !14, !14, !14, !14, !14}
!opencl.ocl.version = !{!15, !15, !15, !15, !15, !15, !15, !15, !15, !15, !15, !15, !15, !15, !15, !15}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, producer: "clang version 19.0.0git (/tmp/llvm/clang 5a5e94265d423fa9eb39dc1b855511195f8dc0fe)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "test.c", directory: "/tmp", checksumkind: CSK_MD5, checksum: "eff61a7cf33c8dd1bd6933250fc90157")
!2 = !{i32 0, i32 64770, i32 466741, !"h", i32 12, i32 0, i32 0}
!3 = !{ptr @__omp_offloading_fd02_71f35_h_l12, !"kernel", i32 1}
!4 = !{i32 1, !"amdhsa_code_object_version", i32 500}
!5 = !{i32 7, !"Dwarf Version", i32 5}
!6 = !{i32 2, !"Debug Info Version", i32 3}
!7 = !{i32 1, !"wchar_size", i32 4}
!8 = !{i32 7, !"openmp", i32 51}
!9 = !{i32 7, !"openmp-device", i32 51}
!10 = !{i32 8, !"PIC Level", i32 2}
!11 = !{i32 7, !"frame-pointer", i32 2}
!12 = !{i32 4, !"amdgpu_hostcall", i32 1}
!13 = !{!"clang version 19.0.0git (/tmp/llvm/clang 5a5e94265d423fa9eb39dc1b855511195f8dc0fe)"}
!14 = !{!"AMD clang version 17.0.0 (https://github.com/RadeonOpenCompute/llvm-project roc-6.0.2 24012 af27734ed982b52a9f1be0f035ac91726fc697e4)"}
!15 = !{i32 2, i32 0}
!16 = distinct !DISubprogram(name: "__omp_offloading_fd02_71f35_h_l12_debug__", scope: !17, file: !17, line: 13, type: !18, scopeLine: 13, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !23)
!17 = !DIFile(filename: "test.c", directory: "/tmp")
!18 = !DISubroutineType(types: !19)
!19 = !{null, !20}
!20 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !21)
!21 = !DIDerivedType(tag: DW_TAG_restrict_type, baseType: !22)
!22 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: null, size: 64)
!23 = !{}
!24 = !DILocalVariable(name: "dyn_ptr", arg: 1, scope: !16, type: !20, flags: DIFlagArtificial)
!25 = !DILocation(line: 0, scope: !16)
!26 = !DILocation(line: 13, column: 3, scope: !16)
!27 = !DILocalVariable(name: "i", scope: !28, file: !17, line: 14, type: !29)
!28 = distinct !DILexicalBlock(scope: !16, file: !17, line: 13, column: 3)
!29 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!30 = !DILocation(line: 14, column: 9, scope: !28)
!31 = !DILocalVariable(name: "a", scope: !28, file: !17, line: 15, type: !32)
!32 = !DICompositeType(tag: DW_TAG_array_type, baseType: !29, size: 64, elements: !33)
!33 = !{!34}
!34 = !DISubrange(count: 2)
!35 = !DILocation(line: 15, column: 9, scope: !28)
!36 = !DILocation(line: 16, column: 5, scope: !28)
!37 = !DILocation(line: 17, column: 5, scope: !28)
!38 = !DILocation(line: 18, column: 3, scope: !28)
!39 = !DILocation(line: 18, column: 3, scope: !16)
!40 = distinct !DISubprogram(name: "__omp_offloading_fd02_71f35_h_l12", scope: !17, file: !17, line: 12, type: !18, scopeLine: 12, flags: DIFlagArtificial | DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !0, retainedNodes: !23)
!41 = !DILocalVariable(name: "dyn_ptr", arg: 1, scope: !40, type: !20, flags: DIFlagArtificial)
!42 = !DILocation(line: 0, scope: !40)
!43 = !DILocation(line: 12, column: 1, scope: !40)
!44 = distinct !DISubprogram(name: "g", scope: !17, file: !17, line: 3, type: !45, scopeLine: 3, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !23)
!45 = !DISubroutineType(types: !46)
!46 = !{null}
!47 = !DILocalVariable(name: "i", scope: !44, file: !17, line: 4, type: !29)
!48 = !DILocation(line: 4, column: 7, scope: !44)
!49 = !DILocalVariable(name: "a", scope: !44, file: !17, line: 5, type: !32)
!50 = !DILocation(line: 5, column: 7, scope: !44)
!51 = !DILocation(line: 6, column: 3, scope: !44)
!52 = !DILocation(line: 7, column: 3, scope: !44)
!53 = !DILocation(line: 8, column: 1, scope: !44)
