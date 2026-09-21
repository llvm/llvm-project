; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; The same DILocalVariable is listed in retainedNodes and referenced by a #dbg_declare.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32T:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[PATH:%[0-9]+]] = OpString "{{[/\\]}}src{{[/\\]}}debug-local-variable-records-and-retained.c"
; CHECK-DAG: [[NAME:%[0-9]+]] = OpString "dup"
; CHECK-DAG: [[INTNAME:%[0-9]+]] = OpString "int"
; The trailing anchors keep e.g. [[C3]] from binding to "OpConstant %3 32".
; CHECK-DAG: [[C0:%[0-9]+]] = OpConstant [[I32T]] 0{{ *$}}
; CHECK-DAG: [[C3:%[0-9]+]] = OpConstant [[I32T]] 3{{ *$}}
; CHECK-DAG: [[DS:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugSource [[PATH]]
; CHECK-DAG: [[INT:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugTypeBasic [[INTNAME]] {{.*}} [[C0]]
; CHECK-DAG: [[DF:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunction {{.*}}
; CHECK-DAG: OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[NAME]] [[INT]] [[DS]] [[C3]] [[C0]] [[DF]] [[C0]]
; CHECK-NOT: DebugLocalVariable

target triple = "spirv64-unknown-unknown"

define spir_func void @f() !dbg !5 {
entry:
  %dup = alloca i32, align 4
    #dbg_declare(ptr %dup, !9, !DIExpression(), !10)
  ret void, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: true, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-local-variable-records-and-retained.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = distinct !DISubprogram(name: "f", linkageName: "f", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0, retainedNodes: !8)
!8 = !{!9}
!9 = !DILocalVariable(name: "dup", scope: !5, file: !1, line: 3, type: !7)
!10 = !DILocation(line: 3, column: 1, scope: !5)
