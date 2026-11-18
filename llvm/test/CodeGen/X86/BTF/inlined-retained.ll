; RUN: llc -mtriple=x86_64-linux-gnu %s -o - | FileCheck %s

; Retained types in both an intermediate call site and the innermost inlined
; function survive. The innermost scope is reached only from a debug record.
; CHECK: .section .BTF
; CHECK: .ascii "middle_value"
; CHECK: .ascii "inner_value"

declare void @sink()

define void @outer() !dbg !4 {
  call void @sink(), !dbg !16
  #dbg_value(i32 0, !17, !DIExpression(), !18)
  ret void, !dbg !15
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}
!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "inline.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 7, !"BTF", i32 1}
!4 = distinct !DISubprogram(name: "outer", scope: !1, file: !1, line: 1, type: !7, unit: !0, spFlags: DISPFlagDefinition)
!5 = distinct !DISubprogram(name: "middle", scope: !1, file: !1, line: 2, type: !7, unit: !0, spFlags: DISPFlagDefinition, retainedNodes: !9)
!6 = distinct !DISubprogram(name: "inner", scope: !1, file: !1, line: 3, type: !7, unit: !0, spFlags: DISPFlagDefinition, retainedNodes: !10)
!7 = !DISubroutineType(types: !8)
!8 = !{null}
!9 = !{!11}
!10 = !{!12}
!11 = !DICompositeType(tag: DW_TAG_enumeration_type, scope: !5, size: 32, elements: !13)
!12 = !DICompositeType(tag: DW_TAG_enumeration_type, scope: !6, size: 32, elements: !14)
!13 = !{!DIEnumerator(name: "middle_value", value: 1)}
!14 = !{!DIEnumerator(name: "inner_value", value: 2)}
!15 = !DILocation(line: 1, scope: !4)
!16 = !DILocation(line: 2, scope: !5, inlinedAt: !15)
!17 = !DILocalVariable(name: "local", scope: !6, type: !19)
!18 = !DILocation(line: 3, scope: !6, inlinedAt: !16)
!19 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
