
; RUN: opt -amdgpu-lower-kernel-calls -S < %s | FileCheck %s
; FIXME: Do we want to be cloning DISubprogram in this case? It currently means
; we produce multiple DIEs for the same source function.
;
; This DISubprogram duplication was added originally as part of
; https://reviews.llvm.org/D32975 which was proposed as a solution to a bug
; where LLVM would produce invalid DIEs with duplicate attributes (see
; https://lists.llvm.org/pipermail/llvm-dev/2017-May/112661.html for
; discussion).
;
; It seems like an alternative to the above change is to define the multiple
; LLVM functions referring to the same DISubprogram as contributing to a
; non-contiguous DW_AT_ranges definition for the same source function:
;
; > A subroutine entry may have either a DW_AT_low_pc and DW_AT_high_pc
; > pair of attributes or a DW_AT_ranges attribute whose values encode the
; > contiguous or non-contiguous address ranges, respectively, of the machine
; > instructions generated for the subroutine (see Section 2.17 on page 51).
;
; To make that change, we would also need to address the semantics of
; retainedNodes, as the two independent LLVM functions could diverge after the
; clone, and updates to one DISubprogram would then affect the other. It may
; be that we could just forbid the updates, or there is a reasonable rule we
; could define to make the updates safe.
;
; There are other related changes, and this seems to stretch back until at
; least 2016 (https://reviews.llvm.org/D33614 https://reviews.llvm.org/D33655
; https://lists.llvm.org/pipermail/llvm-dev/2016-April/098331.html).
;
; Other uses of intra-module cloning;
; * llvm/lib/Transforms/IPO/FunctionSpecialization.cpp - specialises functions with constant parameters
; * llvm/lib/Transforms/IPO/Attributor.cpp - internalization
; * llvm/lib/Transforms/IPO/PartialInlining.cpp
; * llvm/lib/Transforms/Coroutines/CoroSplit.cpp - coroutines
; * llvm/lib/Target/AMDGPU/AMDGPUPropagateAttributes.cpp
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "amdgcn-amd-amdhsa"
@global = global i32 6, !dbg.def !2
; CHECK-LABEL: define amdgpu_kernel void @callee(i32 %I) #0 !dbg ![[#CALLEE_SUBPROGRAM:]]
; CHECK-DAG: call void @llvm.dbg.def(metadata ![[#CALLEE_GLOBAL_BOUNDED_LIFETIME:]], metadata i32 4)
; CHECK-DAG: call void @llvm.dbg.def(metadata ![[#CALLEE_LOCAL_BOUNDED_LIFETIME_I:]], metadata ptr addrspace(5) %I{{.*}})
; CHECK-DAG: call void @llvm.dbg.def(metadata ![[#CALLEE_LOCAL_BOUNDED_LIFETIME_J:]], metadata ptr addrspace(5) %J{{.*}})
define amdgpu_kernel void @callee(i32 %I) #0 !dbg !12 {
entry:
  call void @llvm.dbg.def(metadata !30, metadata i32 4)
  %retval = alloca i32, align 4, addrspace(5)
  %retval.ascast = addrspacecast ptr addrspace(5) %retval to ptr
  %I.addr = alloca i32, align 4, addrspace(5)
  %I.addr.ascast = addrspacecast ptr addrspace(5) %I.addr to ptr
  %J = alloca i32, align 4, addrspace(5)
  %J.ascast = addrspacecast ptr addrspace(5) %J to ptr
  store i32 %I, ptr %I.addr.ascast, align 4
  call void @llvm.dbg.def(metadata !20, metadata ptr addrspace(5) %I.addr)
  call void @llvm.dbg.def(metadata !22, metadata ptr addrspace(5) %J)
  %0 = load i32, ptr %I.addr.ascast, align 4
  store i32 %0, ptr %J.ascast, align 4
  %1 = load i32, ptr %J.ascast, align 4
  ret void
}
; CHECK-LABEL: define hidden void @caller0() #0
; CHECK: call void @__amdgpu_callee_kernel_body(i32 0)
; CHECK-NOT: call amdgpu_kernel void @callee(i32 0)
; CHECK-NOT: call void @llvm.dbg.def{{.*}}
; Function Attrs: nounwind
define hidden void @caller0() #0 !dbg !24 {
entry:
  call amdgpu_kernel void @callee(i32 0) #0, !dbg !27
  ret void
}
; CHECK-LABEL: define hidden void @caller1() #0
; CHECK: call void @__amdgpu_callee_kernel_body(i32 1)
; CHECK-NOT: call amdgpu_kernel void @callee(i32 1)
; CHECK-NOT: call void @llvm.dbg.def{{.*}}
; Function Attrs: nounwind
define hidden void @caller1() #0 !dbg !28 {
entry:
  call amdgpu_kernel void @callee(i32 1) #0, !dbg !29
  ret void
}
; CHECK-LABEL: define internal void @__amdgpu_callee_kernel_body(i32 %I) #0 !dbg ![[#CLONE_SUBPROGRAM:]]
; CHECK-DAG: call void @llvm.dbg.def(metadata ![[#CLONE_GLOBAL_BOUNDED_LIFETIME:]], metadata i32 4)
; CHECK-DAG: call void @llvm.dbg.def(metadata ![[#CLONE_LOCAL_BOUNDED_LIFETIME_I:]], metadata ptr addrspace(5) %I{{.*}})
; CHECK-DAG: call void @llvm.dbg.def(metadata ![[#CLONE_LOCAL_BOUNDED_LIFETIME_J:]], metadata ptr addrspace(5) %J{{.*}})
; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.def(metadata, metadata) #2
; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.kill(metadata) #2
attributes #0 = { nounwind }
; CHECK-LABEL: !llvm.dbg.retainedNodes =
; CHECK-SAME: !{![[#GLOBAL_COMPUTED_LIFETIME:]]}
; CHECK-DAG: ![[#GLOBAL_COMPUTED_LIFETIME]] = distinct !DILifetime(object: ![[#GLOBAL_VARIABLE:]], location: !DIExpr(DIOpArg(0, i32)), argObjects: {![[#GLOBAL_FRAGMENT:]]})
; CHECK-DAG: ![[#GLOBAL_VARIABLE]] = !DIGlobalVariable(name: "global", {{.*}}
; CHECK-DAG: ![[#GLOBAL_FRAGMENT]] = distinct !DIFragment()
; CHECK-DAG: ![[#CALLEE_SUBPROGRAM]] = distinct !DISubprogram(name: "callee", {{.*}}retainedNodes: ![[#CALLEE_RETAINED_NODES:]])
; CHECK-DAG: ![[#CALLEE_RETAINED_NODES]] = !{![[#CALLEE_GLOBAL_COMPUTED_LIFETIME:]], ![[#CALLEE_LOCAL_COMPUTED_LIFETIME:]], ![[#CALLEE_LOCAL_VARIABLE_I:]], ![[#CALLEE_LOCAL_VARIABLE_J:]]}
; CHECK-DAG: ![[#CALLEE_GLOBAL_COMPUTED_LIFETIME]] = distinct !DILifetime(object: ![[#GLOBAL_VARIABLE]], location: !DIExpr(DIOpConstant(i8 2)))
; CHECK-DAG: ![[#CALLEE_LOCAL_COMPUTED_LIFETIME]] = distinct !DILifetime(object: ![[#CALLEE_FRAGMENT:]], location: !DIExpr(DIOpConstant(i8 0)))
; CHECK-DAG: ![[#CALLEE_FRAGMENT]] = distinct !DIFragment()
; CHECK-DAG: ![[#CALLEE_LOCAL_VARIABLE_I]] = !DILocalVariable(name: "I", {{.*}}
; CHECK-DAG: ![[#CALLEE_LOCAL_VARIABLE_J]] = !DILocalVariable(name: "J", {{.*}}
; CHECK-DAG: ![[#CALLEE_GLOBAL_BOUNDED_LIFETIME]] = distinct !DILifetime(object: ![[#GLOBAL_VARIABLE]], location: !DIExpr(DIOpReferrer(i32)))
; CHECK-DAG: ![[#CLONE_SUBPROGRAM]] = distinct !DISubprogram(name: "callee", {{.*}}retainedNodes: ![[#CLONE_RETAINED_NODES:]])
; CHECK-DAG: ![[#CLONE_RETAINED_NODES]] = !{![[#CLONE_GLOBAL_COMPUTED_LIFETIME:]], ![[#CLONE_LOCAL_COMPUTED_LIFETIME:]], ![[#CLONE_LOCAL_VARIABLE_I:]], ![[#CLONE_LOCAL_VARIABLE_J:]]}
; CHECK-DAG: ![[#CLONE_GLOBAL_COMPUTED_LIFETIME]] = distinct !DILifetime(object: ![[#GLOBAL_VARIABLE]], location: !DIExpr(DIOpConstant(i8 2)))
; CHECK-DAG: ![[#CLONE_LOCAL_COMPUTED_LIFETIME]] = distinct !DILifetime(object: ![[#CLONE_FRAGMENT:]], location: !DIExpr(DIOpConstant(i8 0)))
; CHECK-DAG: ![[#CLONE_FRAGMENT]] = distinct !DIFragment()
; CHECK-DAG: ![[#CLONE_LOCAL_VARIABLE_I]] = !DILocalVariable(name: "I", {{.*}}
; CHECK-DAG: ![[#CLONE_LOCAL_VARIABLE_J]] = !DILocalVariable(name: "J", {{.*}}
; CHECK-DAG: ![[#CLONE_GLOBAL_BOUNDED_LIFETIME]] = distinct !DILifetime(object: ![[#GLOBAL_VARIABLE]], location: !DIExpr(DIOpReferrer(i32)))
!llvm.dbg.retainedNodes = !{!0}
!llvm.dbg.cu = !{!3}
!llvm.module.flags = !{!6, !7, !8, !9, !10}
!llvm.ident = !{!11}
!0 = distinct !DILifetime(object: !1, location: !DIExpr(DIOpArg(0, i32)), argObjects: {!2})
!1 = !DIGlobalVariable(name: "global", scope: !3, type: !15, isLocal: false, isDefinition: true)
!2 = distinct !DIFragment()
!3 = distinct !DICompileUnit(language: DW_LANG_C_plus_plus_11, file: !4, producer: "clang version 13.0.0 (https://github.com/llvm/llvm-project.git 87656a3134c7c03565efca85352a58541ce68789)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !5, imports: !5, splitDebugInlining: false, nameTableKind: None)
!4 = !DIFile(filename: "kernelCalls.cl", directory: "/rocm-gdb-symbols")
!5 = !{}
!6 = !{i32 7, !"Dwarf Version", i32 4}
!7 = !{i32 2, !"Debug Info Version", i32 4}
!8 = !{i32 1, !"wchar_size", i32 4}
!9 = !{i32 7, !"PIC Level", i32 1}
!10 = !{i32 7, !"frame-pointer", i32 2}
!11 = !{!"clang version 13.0.0 (https://github.com/llvm/llvm-project.git 87656a3134c7c03565efca85352a58541ce68789)"}
!12 = distinct !DISubprogram(name: "callee", scope: !3, file: !4, line: 1, type: !13, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !3, retainedNodes: !16)
!13 = !DISubroutineType(types: !14)
!14 = !{!15, !15}
!15 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!16 = !{!17, !18, !21, !23}
!17 = distinct !DILifetime(object: !1, location: !DIExpr(DIOpConstant(i8 2)))
!18 = distinct !DILifetime(object: !19, location: !DIExpr(DIOpConstant(i8 0)))
!19 = distinct !DIFragment()
!20 = distinct !DILifetime(object: !21, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i32)))
!21 = !DILocalVariable(name: "I", arg: 1, scope: !12, file: !4, line: 1, type: !15)
!22 = distinct !DILifetime(object: !23, location: !DIExpr(DIOpReferrer(ptr addrspace(5)), DIOpDeref(i32)), argObjects: {!21, !19})
!23 = !DILocalVariable(name: "J", scope: !12, file: !4, line: 1, type: !15)
!24 = distinct !DISubprogram(name: "caller0", scope: !3, file: !4, line: 2, type: !25, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !3, retainedNodes: !5)
!25 = !DISubroutineType(types: !26)
!26 = !{!15}
!27 = !DILocation(line: 2, column: 2, scope: !24)
!28 = distinct !DISubprogram(name: "caller1", scope: !3, file: !4, line: 2, type: !25, scopeLine: 2, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !3, retainedNodes: !5)
!29 = !DILocation(line: 2, column: 2, scope: !28)
!30 = distinct !DILifetime(object: !1, location: !DIExpr(DIOpReferrer(i32)))
