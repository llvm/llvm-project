; RUN: llc %s --stop-before=finalize-isel -o - | FileCheck %s --implicit-check-not=DBG_VALUE

;; Check that when a debug value is invalidated during Instruction Selection,
;; we produce an undef DBG_VALUE/DBG_VALUE_LIST.

; CHECK-LABEL: body:

; CHECK: DBG_VALUE $noreg, $noreg, ![[VAR:[0-9]+]]

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

@intel_pmu_enable_bts_config = external dso_local local_unnamed_addr global i32, align 4, !dbg !0

define dso_local i32 @intel_pmu_enable_bts() local_unnamed_addr !dbg !16 {
entry:
  %0 = extractvalue { i32, i64 } zeroinitializer, 1
  %1 = load i32, i32* @intel_pmu_enable_bts_config, align 4
  call void @llvm.dbg.value(metadata !DIArgList(i64 %0, i32 %1), metadata !20, metadata !DIExpression(DW_OP_LLVM_arg, 0, DW_OP_LLVM_arg, 1, DW_OP_or, DW_OP_stack_value)), !dbg !23
  ret i32 %1
}

declare void @llvm.dbg.value(metadata, metadata, metadata)

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!14, !15}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "intel_pmu_enable_bts_config", scope: !2, file: !3, line: 1, type: !8, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 13.0.0", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, globals: !5, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "invalidated-dbg-value-is-undef.ll", directory: "/")
!4 = !{}
!5 = !{!0, !6, !9, !11}
!6 = !DIGlobalVariableExpression(var: !7, expr: !DIExpression())
!7 = distinct !DIGlobalVariable(name: "intel_pmu_enable_bts___trans_tmp_1", scope: !2, file: !3, line: 1, type: !8, isLocal: false, isDefinition: true)
!8 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!9 = !DIGlobalVariableExpression(var: !10, expr: !DIExpression())
!10 = distinct !DIGlobalVariable(name: "intel_pmu_enable_bts___ecx", scope: !2, file: !3, line: 2, type: !8, isLocal: false, isDefinition: true)
!11 = !DIGlobalVariableExpression(var: !12, expr: !DIExpression())
!12 = distinct !DIGlobalVariable(name: "intel_pmu_enable_bts___eax", scope: !2, file: !3, line: 3, type: !13, isLocal: false, isDefinition: true)
!13 = !DIBasicType(name: "long int", size: 64, encoding: DW_ATE_signed)
!14 = !{i32 2, !"Debug Info Version", i32 3}
!15 = !{i32 7, !"uwtable", i32 1}
!16 = distinct !DISubprogram(name: "intel_pmu_enable_bts", scope: !3, file: !3, line: 4, type: !17, scopeLine: 4, flags: DIFlagAllCallsDescribed, spFlags: DISPFlagDefinition | DISPFlagOptimized, unit: !2, retainedNodes: !19)
!17 = !DISubroutineType(types: !18)
!18 = !{!8}
!19 = !{!20, !21}
!20 = !DILocalVariable(name: "debugctlmsr", scope: !16, file: !3, line: 5, type: !13)
!21 = !DILocalVariable(name: "low", scope: !16, file: !3, line: 17, type: !22)
!22 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!23 = !DILocation(line: 0, scope: !16)
