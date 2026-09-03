; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; The scope changes while the source position stays identical.
; At the moment, the emission of DebugLine depends on the DILocation pointer
; which leads to an additional DebugLine being emitted.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[PATH:%[0-9]+]] = OpString "{{[/\\]}}src{{[/\\]}}debug-scope-same-line.c"
; CHECK-DAG: [[DS:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugSource [[PATH]]
; CHECK-DAG: [[DF:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunction {{.*}}
; CHECK-DAG: [[LB:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLexicalBlock [[DS]] {{.*}} [[DF]]
; CHECK-DAG: [[V3:%[0-9]+]] = OpConstant [[I32]] 3{{$}}
; CHECK-DAG: [[V4:%[0-9]+]] = OpConstant [[I32]] 4{{$}}
; CHECK-DAG: [[V5:%[0-9]+]] = OpConstant [[I32]] 5{{$}}
; CHECK-DAG: [[V6:%[0-9]+]] = OpConstant [[I32]] 6{{$}}

; CHECK:      [[FN:%[0-9]+]] = OpFunction
; CHECK-NEXT: [[N:%[0-9]+]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugFunctionDefinition [[DF]] [[FN]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugScope [[DF]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V3]] [[V3]] [[V5]] [[V6]]
; CHECK-NEXT: [[A:%[0-9]+]] = OpIAdd [[I32]]

; Scope-only change: new DebugScope, plus a DebugLine identical to the one above.
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugScope [[LB]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V3]] [[V3]] [[V5]] [[V6]]
; CHECK-NEXT: [[B:%[0-9]+]] = OpIAdd [[I32]]

; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugScope [[DF]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V4]] [[V4]] [[V3]] [[V4]]
; CHECK-NEXT: OpReturnValue [[B]]
; CHECK-NEXT: OpFunctionEnd

target triple = "spirv64-unknown-unknown"

define spir_func i32 @same_line(i32 %n) !dbg !5 {
entry:
  %a = add i32 %n, 1, !dbg !9    ; function scope, line 3, col 5
  %b = add i32 %a, 1, !dbg !11   ; lexical block, line 3, col 5 -- scope only
  ret i32 %b, !dbg !12           ; back to function scope, line 4, col 3
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-scope-same-line.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}

!4 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!5 = distinct !DISubprogram(name: "same_line", linkageName: "same_line", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DILocation(line: 3, column: 5, scope: !5)
!10 = distinct !DILexicalBlock(scope: !5, file: !1, line: 2, column: 6)
!11 = !DILocation(line: 3, column: 5, scope: !10)
!12 = !DILocation(line: 4, column: 3, scope: !5)
