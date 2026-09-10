; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[PATH:%[0-9]+]] = OpString "{{[/\\]}}src{{[/\\]}}debug-scope.c"
; CHECK-DAG: [[DS:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugSource [[PATH]]
; CHECK-DAG: [[DF:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunction {{.*}}
; CHECK-DAG: [[LB:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLexicalBlock [[DS]] {{.*}} [[DF]]
; CHECK-DAG: [[V2:%[0-9]+]] = OpConstant [[I32]] 2{{$}}
; CHECK-DAG: [[V3:%[0-9]+]] = OpConstant [[I32]] 3{{$}}
; CHECK-DAG: [[V4:%[0-9]+]] = OpConstant [[I32]] 4{{$}}
; CHECK-DAG: [[V5:%[0-9]+]] = OpConstant [[I32]] 5{{$}}
; CHECK-DAG: [[V7:%[0-9]+]] = OpConstant [[I32]] 7{{$}}
; CHECK-DAG: [[V10:%[0-9]+]] = OpConstant [[I32]] 10{{$}}
; CHECK-DAG: [[V11:%[0-9]+]] = OpConstant [[I32]] 11{{$}}

; CHECK:      [[FN:%[0-9]+]] = OpFunction
; CHECK-NEXT: [[N:%[0-9]+]] = OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugFunctionDefinition [[DF]] [[FN]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugScope [[DF]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V2]] [[V2]] [[V10]] [[V11]]
; CHECK-NEXT: [[A:%[0-9]+]] = OpIAdd [[I32]] [[N]]
; CHECK-NOT:  OpExtInst [[VOID]] [[EXT]] DebugScope
; CHECK-NOT:  OpExtInst [[VOID]] [[EXT]] DebugLine
; CHECK-NEXT: [[B:%[0-9]+]] = OpIAdd [[I32]] [[A]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugScope [[LB]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V5]] [[V5]] [[V3]] [[V4]]
; CHECK-NEXT: [[C:%[0-9]+]] = OpIAdd [[I32]] [[B]]
; CHECK-NOT:  OpExtInst [[VOID]] [[EXT]] DebugScope
; CHECK-NOT:  OpExtInst [[VOID]] [[EXT]] DebugLine
; CHECK-NEXT: [[D:%[0-9]+]] = OpIAdd [[I32]] [[C]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugScope [[DF]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V7]] [[V7]] [[V3]] [[V4]]
; CHECK-NEXT: OpReturnValue [[D]]
; CHECK-NEXT: OpFunctionEnd

target triple = "spirv64-unknown-unknown"

define spir_func i32 @scoped(i32 %n) !dbg !5 {
entry:
  %a = add i32 %n, 1, !dbg !9    ; function scope, line 2, col 10
  %b = add i32 %a, 1, !dbg !9    ; same scope+loc -> dedup
  %c = add i32 %b, 1, !dbg !11   ; nested lexical block scope, line 5, col 3
  %d = add i32 %c, 1, !dbg !11   ; same scope+loc -> dedup
  ret i32 %d, !dbg !13           ; back to function scope, line 7, col 3
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-scope.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}

!4 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!5 = distinct !DISubprogram(name: "scoped", linkageName: "scoped", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DILocation(line: 2, column: 10, scope: !5)
!10 = distinct !DILexicalBlock(scope: !5, file: !1, line: 4, column: 5)
!11 = !DILocation(line: 5, column: 3, scope: !10)
!13 = !DILocation(line: 7, column: 3, scope: !5)
