; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; A declare has a location of its own, so it takes part in DebugLine and
; DebugScope tracking like a real instruction. The body interleaves declares
; and instructions whose locations disagree on purpose:
;
;   #dbg_declare(ptr %x, !9, !DIExpression(), !20)  ; !20 is line 20
;   store i32 1, ptr %x, align 4, !dbg !21          ; !21 is line 5
;

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[PATH:%[0-9]+]] = OpString "{{[/\\]}}src{{[/\\]}}debug-declare-line-scope.c"
; CHECK-DAG: [[X:%[0-9]+]] = OpString "x"
; CHECK-DAG: [[Y:%[0-9]+]] = OpString "y"
; CHECK-DAG: [[DS:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugSource [[PATH]]
; CHECK-DAG: [[DF:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunction {{.*}}
; CHECK-DAG: [[LB:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLexicalBlock [[DS]] {{.*}} [[DF]]
; CHECK-DAG: [[XVAR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[X]]
; CHECK-DAG: [[YVAR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[Y]]
; CHECK-DAG: [[EXPR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugExpression{{ *$}}
; CHECK-DAG: [[V3:%[0-9]+]] = OpConstant [[I32]] 3{{$}}
; CHECK-DAG: [[V4:%[0-9]+]] = OpConstant [[I32]] 4{{$}}
; CHECK-DAG: [[V5:%[0-9]+]] = OpConstant [[I32]] 5{{$}}
; CHECK-DAG: [[V6:%[0-9]+]] = OpConstant [[I32]] 6{{$}}
; CHECK-DAG: [[V7:%[0-9]+]] = OpConstant [[I32]] 7{{$}}
; CHECK-DAG: [[V8:%[0-9]+]] = OpConstant [[I32]] 8{{$}}
; CHECK-DAG: [[V9:%[0-9]+]] = OpConstant [[I32]] 9{{$}}
; CHECK-DAG: [[V10:%[0-9]+]] = OpConstant [[I32]] 10{{$}}
; CHECK-DAG: [[V20:%[0-9]+]] = OpConstant [[I32]] 20{{$}}
; CHECK-DAG: [[V31:%[0-9]+]] = OpConstant [[I32]] 31{{$}}

; CHECK:      [[FN:%[0-9]+]] = OpFunction
; CHECK-NEXT: OpLabel
; CHECK-NEXT: [[XADDR:%[0-9]+]] = OpVariable {{%[0-9]+}} Function
; CHECK-NEXT: [[YADDR:%[0-9]+]] = OpVariable {{%[0-9]+}} Function
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugFunctionDefinition [[DF]] [[FN]]

; The declare's own location, line 20, not the line 5 of the store below it.
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugScope [[DF]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V20]] [[V20]] [[V7]] [[V8]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugDeclare [[XVAR]] [[XADDR]] [[EXPR]]
; CHECK-NEXT: ;DEBUG_VALUE:

; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V5]] [[V5]] [[V3]] [[V4]]
; CHECK-NEXT: OpStore [[XADDR]]

; A declare in a lexical block moves the scope, on the declare alone.
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugScope [[LB]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V31]] [[V31]] [[V9]] [[V10]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugDeclare [[YVAR]] [[YADDR]] [[EXPR]]
; CHECK-NEXT: ;DEBUG_VALUE:

; The unlocated store drops both, rather than keeping the declare's line.
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugNoScope
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugNoLine
; CHECK-NEXT: OpStore [[YADDR]]

; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugScope [[DF]]
; CHECK-NEXT: OpExtInst [[VOID]] [[EXT]] DebugLine [[DS]] [[V6]] [[V6]] [[V3]] [[V4]]
; CHECK-NEXT: OpReturn
; CHECK-NEXT: OpFunctionEnd

target triple = "spirv64-unknown-unknown"

define spir_func void @f() !dbg !5 {
entry:
  %x = alloca i32, align 4
  %y = alloca i32, align 4
    #dbg_declare(ptr %x, !9, !DIExpression(), !20)  ; function scope, line 20
  store i32 1, ptr %x, align 4, !dbg !21            ; function scope, line 5
    #dbg_declare(ptr %y, !10, !DIExpression(), !22) ; lexical block, line 31
  store i32 2, ptr %y, align 4                      ; no debug location
  ret void, !dbg !23                                ; function scope, line 6
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-declare-line-scope.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}

!4 = !DISubroutineType(types: !6)
!6 = !{null}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!5 = distinct !DISubprogram(name: "f", linkageName: "f", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!8 = distinct !DILexicalBlock(scope: !5, file: !1, line: 30, column: 3)
!9 = !DILocalVariable(name: "x", scope: !5, file: !1, line: 20, type: !7)
!10 = !DILocalVariable(name: "y", scope: !8, file: !1, line: 31, type: !7)
!20 = !DILocation(line: 20, column: 7, scope: !5)
!21 = !DILocation(line: 5, column: 3, scope: !5)
!22 = !DILocation(line: 31, column: 9, scope: !8)
!23 = !DILocation(line: 6, column: 3, scope: !5)
