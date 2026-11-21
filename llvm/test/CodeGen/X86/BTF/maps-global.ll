; RUN: llc -mtriple=x86_64-linux-gnu %s -o - | FileCheck %s
; CHECK: BTF_KIND_VAR
; CHECK: BTF_KIND_DATASEC
; CHECK: .byte 97
; CHECK: .ascii ".maps"
@a = global i32 1, section ".maps", !dbg !0
!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!6, !7, !8}
!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "a", scope: !2, file: !3, line: 1, type: !4, isLocal: false, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C11, file: !3, emissionKind: FullDebug, globals: !5)
!3 = !DIFile(filename: "test.c", directory: "/")
!4 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!5 = !{!0}
!6 = !{i32 7, !"Dwarf Version", i32 5}
!7 = !{i32 2, !"Debug Info Version", i32 3}
!8 = !{i32 7, !"BTF", i32 1}
