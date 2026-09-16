; asm-verbose=0 keeps AsmPrinter's ;DEBUG_VALUE: comments out of the output so
; the CHECK-NEXT chain tracks SPIR-V instructions only.
; RUN: llc --verify-machineinstrs -O0 -mtriple=spirv64-unknown-unknown --asm-verbose=0 --spirv-ext=+SPV_KHR_non_semantic_info %s -o - | FileCheck %s
; RUN: %if spirv-tools %{ llc --verify-machineinstrs --spirv-ext=+SPV_KHR_non_semantic_info -O0 -mtriple=spirv64-unknown-unknown %s -o - -filetype=obj | spirv-val %}

; The value folds to a constant, which module analysis hoists into
; MB_TypeConstVars and marks skip-emission where it was. That section is
; written before every function body, so the id is available to a record that
; names it even though nothing in the body defines it.

; CHECK-DAG: [[EXT:%[0-9]+]] = OpExtInstImport "NonSemantic.Shader.DebugInfo.100"
; CHECK-DAG: [[VOID:%[0-9]+]] = OpTypeVoid
; CHECK-DAG: [[I32:%[0-9]+]] = OpTypeInt 32 0
; CHECK-DAG: [[NAME:%[0-9]+]] = OpString "folded"
; CHECK-DAG: [[VAR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugLocalVariable [[NAME]]
; CHECK-DAG: [[EXPR:%[0-9]+]] = OpExtInst [[VOID]] [[EXT]] DebugExpression{{ *$}}

; The binding is emitted at its own position, ahead of the only instruction in
; the body that uses the constant.
; CHECK: OpFunction
; CHECK: OpExtInst [[VOID]] [[EXT]] DebugValue [[VAR]] [[C3:%[0-9]+]] [[EXPR]]{{ *$}}
; CHECK-NEXT: OpIAdd [[I32]] {{%[0-9]+}} [[C3]]

target triple = "spirv64-unknown-unknown"

define spir_func i32 @f(i32 %x) !dbg !5 {
entry:
  %c = add i32 1, 2, !dbg !10
    #dbg_value(i32 %c, !9, !DIExpression(), !10)
  %r = add i32 %x, %c, !dbg !10
  ret i32 %r, !dbg !10
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, producer: "clang", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, splitDebugInlining: false, nameTableKind: None)
!1 = !DIFile(filename: "debug-value-module-scope.c", directory: "/src")
!2 = !{i32 7, !"Dwarf Version", i32 5}
!3 = !{i32 2, !"Debug Info Version", i32 3}
!4 = !DISubroutineType(types: !6)
!6 = !{!7, !7}
!7 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = distinct !DISubprogram(name: "f", linkageName: "f", scope: !1, file: !1, line: 1, type: !4, scopeLine: 1, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !0)
!9 = !DILocalVariable(name: "folded", scope: !5, file: !1, line: 2, type: !7)
!10 = !DILocation(line: 3, column: 3, scope: !5)
