; RUN: opt -dropped-variable-stats -passes=simplifycfg -disable-output %s | FileCheck %s

; Each inlined instance of a variable is tracked separately. g is inlined twice
; into each function, at call sites in the same scope. SimplifyCFG deletes the
; unreachable block holding the dropped #dbg_values.

; CHECK:      Pass Level, Pass Name, Num of Dropped Variables, Func or Module Name

; y inlined at !100 is dropped, y inlined at !102 is kept.
define void @one_dropped() !dbg !10 {
; CHECK-NEXT: Function, SimplifyCFGPass, 1, one_dropped
entry:
  call void @ext(), !dbg !101                        ; inlinedAt: !100
    #dbg_value(i32 0, !22, !DIExpression(), !103)    ; inlinedAt: !102
  call void @ext(), !dbg !104                        ; inlinedAt: !102
  ret void
dead:
    #dbg_value(i32 7, !22, !DIExpression(), !105)    ; inlinedAt: !100
  ret void
}

; y inlined at !112 is dropped, y inlined at !110 is dead.
define void @both_dropped_one_live() !dbg !11 {
; CHECK-NEXT: Function, SimplifyCFGPass, 1, both_dropped_one_live
entry:
  call void @ext(), !dbg !114                        ; inlinedAt: !112
  ret void
dead:
    #dbg_value(i32 7, !22, !DIExpression(), !115)    ; inlinedAt: !110
    #dbg_value(i32 7, !22, !DIExpression(), !113)    ; inlinedAt: !112
  ret void
}

; y inlined at !120 and at !122 are both dropped.
define void @both_dropped_both_live() !dbg !12 {
; CHECK-NEXT: Function, SimplifyCFGPass, 2, both_dropped_both_live
entry:
  call void @ext(), !dbg !121                        ; inlinedAt: !120
  call void @ext(), !dbg !124                        ; inlinedAt: !122
  ret void
dead:
    #dbg_value(i32 7, !22, !DIExpression(), !125)    ; inlinedAt: !120
    #dbg_value(i32 7, !22, !DIExpression(), !123)    ; inlinedAt: !122
  ret void
}

; CHECK-NOT:  {{.}}

declare void @ext()

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}
!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/tmp")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DISubroutineType(types: !4)
!4 = !{null}
!5 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = distinct !DISubprogram(name: "one_dropped", scope: !1, file: !1, line: 1, type: !3, spFlags: DISPFlagDefinition, unit: !0)
!11 = distinct !DISubprogram(name: "both_dropped_one_live", scope: !1, file: !1, line: 1, type: !3, spFlags: DISPFlagDefinition, unit: !0)
!12 = distinct !DISubprogram(name: "both_dropped_both_live", scope: !1, file: !1, line: 1, type: !3, spFlags: DISPFlagDefinition, unit: !0)
!20 = distinct !DISubprogram(name: "g", scope: !1, file: !1, line: 10, type: !3, spFlags: DISPFlagDefinition, unit: !0)
!22 = !DILocalVariable(name: "y", scope: !20, file: !1, line: 11, type: !5)

!100 = distinct !DILocation(line: 3, scope: !10)
!101 = !DILocation(line: 12, scope: !20, inlinedAt: !100)
!102 = distinct !DILocation(line: 4, scope: !10)
!103 = !DILocation(line: 11, scope: !20, inlinedAt: !102)
!104 = !DILocation(line: 12, scope: !20, inlinedAt: !102)
!105 = !DILocation(line: 11, scope: !20, inlinedAt: !100)

!110 = distinct !DILocation(line: 3, scope: !11)
!112 = distinct !DILocation(line: 4, scope: !11)
!113 = !DILocation(line: 11, scope: !20, inlinedAt: !112)
!114 = !DILocation(line: 12, scope: !20, inlinedAt: !112)
!115 = !DILocation(line: 11, scope: !20, inlinedAt: !110)

!120 = distinct !DILocation(line: 3, scope: !12)
!121 = !DILocation(line: 12, scope: !20, inlinedAt: !120)
!122 = distinct !DILocation(line: 4, scope: !12)
!123 = !DILocation(line: 11, scope: !20, inlinedAt: !122)
!124 = !DILocation(line: 12, scope: !20, inlinedAt: !122)
!125 = !DILocation(line: 11, scope: !20, inlinedAt: !120)
