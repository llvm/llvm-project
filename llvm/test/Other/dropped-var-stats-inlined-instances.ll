; RUN: opt -dropped-variable-stats -passes=simplifycfg -disable-output %s | FileCheck %s

; SimplifyCFG deletes the unreachable block holding the dropped #dbg_values.

; CHECK:      Pass Level, Pass Name, Num of Dropped Variables, Func or Module Name

; The following tests check that each inlined instance of a variable is tracked
; separately. g is inlined twice into each function, at call sites in the same scope.

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

; The following tests check that scopes are traversed correctly.

; The remaining instruction has two visible frames (!131, !130).
; The variable at !132 has the same scope as !130, so it is still visible and dropped.
define void @inlined_code_in_scope() !dbg !13 {
; CHECK-NEXT: Function, SimplifyCFGPass, 1, inlined_code_in_scope
entry:
  call void @ext(), !dbg !131                        ; inlinedAt: !130
  ret void
dead:
    #dbg_value(i32 7, !21, !DIExpression(), !132)    ; not inlined
  ret void
}

; The remaining instruction has three visible frames (!143, !142, !140).
; The variable at !141 has the same inlinedAt as !142, so it is still visible and dropped.
define void @nested_inlined_code_in_scope() !dbg !14 {
; CHECK-NEXT: Function, SimplifyCFGPass, 1, nested_inlined_code_in_scope
entry:
  call void @ext(), !dbg !143                        ; inlinedAt: !142 (inlinedAt: !140)
  ret void
dead:
    #dbg_value(i32 7, !22, !DIExpression(), !141)    ; inlinedAt: !140
  ret void
}

; The remaining instruction has two visible frames (!154, !153).
; The variable at !155 is in block !151, a sibling block of !152 (the scope of !153), so
; it is not visible, and is not considered dropped.
define void @inlined_code_in_sibling_scope() !dbg !15 {
entry:
  call void @ext(), !dbg !154                        ; inlinedAt: !153
  ret void
dead:
    #dbg_value(i32 7, !23, !DIExpression(), !155)
  ret void
}

; One remaining instruction is in a parent scope, one is in a sibling scope to the
; variable. The variable is not visible from them, and is not considered dropped.
define void @code_in_parent_and_sibling_scope() !dbg !16 {
entry:
  call void @ext(), !dbg !163                        ; scope: !16
  call void @ext(), !dbg !164                        ; scope: !162 (parent: !16)
  ret void
dead:
    #dbg_value(i32 7, !24, !DIExpression(), !165)    ; scope: !161 (parent: !16)
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
!13 = distinct !DISubprogram(name: "inlined_code_in_scope", scope: !1, file: !1, line: 1, type: !3, spFlags: DISPFlagDefinition, unit: !0)
!14 = distinct !DISubprogram(name: "nested_inlined_code_in_scope", scope: !1, file: !1, line: 1, type: !3, spFlags: DISPFlagDefinition, unit: !0)
!15 = distinct !DISubprogram(name: "inlined_code_in_sibling_scope", scope: !1, file: !1, line: 1, type: !3, spFlags: DISPFlagDefinition, unit: !0)
!16 = distinct !DISubprogram(name: "code_in_parent_and_sibling_scope", scope: !1, file: !1, line: 1, type: !3, spFlags: DISPFlagDefinition, unit: !0)
!20 = distinct !DISubprogram(name: "g", scope: !1, file: !1, line: 10, type: !3, spFlags: DISPFlagDefinition, unit: !0)
!30 = distinct !DISubprogram(name: "h", scope: !1, file: !1, line: 20, type: !3, spFlags: DISPFlagDefinition, unit: !0)
!21 = !DILocalVariable(name: "x", scope: !13, file: !1, line: 2, type: !5)
!22 = !DILocalVariable(name: "y", scope: !20, file: !1, line: 11, type: !5)
!23 = !DILocalVariable(name: "z", scope: !151, file: !1, line: 3, type: !5)
!24 = !DILocalVariable(name: "w", scope: !161, file: !1, line: 3, type: !5)

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

!130 = distinct !DILocation(line: 3, scope: !13)
!131 = !DILocation(line: 12, scope: !20, inlinedAt: !130)
!132 = !DILocation(line: 2, scope: !13)

!140 = distinct !DILocation(line: 3, scope: !14)
!141 = !DILocation(line: 11, scope: !20, inlinedAt: !140)
!142 = distinct !DILocation(line: 12, scope: !20, inlinedAt: !140)
!143 = !DILocation(line: 21, scope: !30, inlinedAt: !142)

!151 = distinct !DILexicalBlock(scope: !15, file: !1, line: 2)
!152 = distinct !DILexicalBlock(scope: !15, file: !1, line: 5)
!153 = distinct !DILocation(line: 6, scope: !152)
!154 = !DILocation(line: 12, scope: !20, inlinedAt: !153)
!155 = !DILocation(line: 3, scope: !151)

!161 = distinct !DILexicalBlock(scope: !16, file: !1, line: 2)
!162 = distinct !DILexicalBlock(scope: !16, file: !1, line: 5)
!163 = !DILocation(line: 1, scope: !16)
!164 = !DILocation(line: 6, scope: !162)
!165 = !DILocation(line: 3, scope: !161)
