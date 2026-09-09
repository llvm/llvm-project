; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Exercise the implicit close of a DebugScope region at a basic block boundary,
; for a lexical block that spans two blocks.
;
; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[PATH:%[0-9]+]] = OpString "{{[/\\]}}src{{[/\\]}}debug-scope-block.c"
; CHECK-DAG: [[DS:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugSource [[PATH]]
; CHECK-DAG: [[DF:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunction {{.*}}
; CHECK-DAG: [[LB:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLexicalBlock [[DS]] {{.*}} [[DF]]
; CHECK-DAG: [[V3:%[0-9]+]] = OpConstant [[I32]] 3{{$}}
; CHECK-DAG: [[V4:%[0-9]+]] = OpConstant [[I32]] 4{{$}}
; CHECK-DAG: [[V5:%[0-9]+]] = OpConstant [[I32]] 5{{$}}
; CHECK-DAG: [[V6:%[0-9]+]] = OpConstant [[I32]] 6{{$}}
; CHECK-DAG: [[V7:%[0-9]+]] = OpConstant [[I32]] 7{{$}}

; CHECK:      [[FN:%[0-9]+]] = OpFunction
; CHECK-NEXT: OpFunctionParameter
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugFunctionDefinition [[DF]] [[FN]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugScope [[LB]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V4]] [[V4]] [[V5]] [[V6]]
; CHECK-NEXT: [[T0:%[0-9]+]] = OpIAdd [[I32]]
; CHECK-NEXT: OpBranch

; then: both regions reopen even though neither scope nor position changed.
; CHECK-NEXT: OpLabel
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugScope [[LB]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V4]] [[V4]] [[V5]] [[V6]]
; CHECK-NEXT: [[T1:%[0-9]+]] = OpIMul [[I32]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugScope [[DF]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V7]] [[V7]] [[V3]] [[V4]]
; CHECK-NEXT: OpReturnValue [[T1]]
; CHECK-NEXT: OpFunctionEnd

target triple = "spirv64-unknown-unknown"

define spir_func i32 @block_scope(i32 %x) !dbg !5 {
entry:
  %t0 = add i32 %x, 1, !dbg !9     ; lexical block, line 4, col 5
  br label %then, !dbg !9          ; same scope and position as the add

then:
  %t1 = mul i32 %t0, %t0, !dbg !9  ; same scope and position, new block
  ret i32 %t1, !dbg !12            ; back to function scope, line 7, col 3
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-scope-block.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}

!4 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!5 = distinct !DISubprogram(name: "block_scope", linkageName: "block_scope", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!10 = distinct !DILexicalBlock(scope: !5, file: !1, line: 3, column: 5)
!9 = !DILocation(line: 4, column: 5, scope: !10)
!12 = !DILocation(line: 7, column: 3, scope: !5)
