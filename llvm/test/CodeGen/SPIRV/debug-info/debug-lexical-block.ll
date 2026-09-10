; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; Exercise NonSemantic DebugLexicalBlock for two levels of DILexicalBlock
; nesting. Scope tree on the left, emitted NSDI instruction on the right:
;
;   !5  DISubprogram "nested"      -> DebugFunction     [[DF]]
;   `- !8  DILexicalBlock 42:43    -> DebugLexicalBlock [[OUTER]]
;      `- !10 DILexicalBlock 44:45 -> DebugLexicalBlock [[INNER]]
;
; Each block's Parent operand is the instruction emitted for its parent scope,
; so [[OUTER]] points at [[DF]] and [[INNER]] points at [[OUTER]].

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[PATH:%[0-9]+]] = OpString "{{[/\\]}}src{{[/\\]}}debug-lexical-block.c"
; CHECK-DAG: [[NAME:%[0-9]+]] = OpString "nested"
; CHECK-DAG: [[C42:%[0-9]+]] = OpConstant [[I32]] 42
; CHECK-DAG: [[C43:%[0-9]+]] = OpConstant [[I32]] 43
; CHECK-DAG: [[C44:%[0-9]+]] = OpConstant [[I32]] 44
; CHECK-DAG: [[C45:%[0-9]+]] = OpConstant [[I32]] 45
; CHECK-DAG: [[DS:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugSource [[PATH]]
; CHECK-DAG: [[DF:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugFunction [[NAME]] {{.*}} [[DS]] {{.*}}
; CHECK-DAG: [[OUTER:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLexicalBlock [[DS]] [[C42]] [[C43]] [[DF]]
; CHECK-DAG: [[INNER:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLexicalBlock [[DS]] [[C44]] [[C45]] [[OUTER]]

target triple = "spirv64-unknown-unknown"

define spir_func i32 @nested(i32 %n) !dbg !5 {
entry:
  %n.addr = alloca i32, align 4
  store i32 %n, ptr %n.addr, align 4
  br label %outer, !dbg !9

outer:
  br label %inner, !dbg !11

inner:
  %v = load i32, ptr %n.addr, align 4, !dbg !13
  ret i32 %v, !dbg !13
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-lexical-block.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}

!4 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)

!5 = distinct !DISubprogram(name: "nested", linkageName: "nested", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!8 = distinct !DILexicalBlock(scope: !5, file: !1, line: 42, column: 43)
!9 = !DILocation(line: 42, column: 43, scope: !8)
!10 = distinct !DILexicalBlock(scope: !8, file: !1, line: 44, column: 45)
!11 = !DILocation(line: 44, column: 45, scope: !10)
!13 = !DILocation(line: 46, column: 47, scope: !10)
