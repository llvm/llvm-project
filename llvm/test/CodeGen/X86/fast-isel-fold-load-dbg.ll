; RUN: llc -O0 -mtriple=x86_64-unknown-unknown -stop-after=finalize-isel %s -o - | \
; RUN:    FileCheck %s --check-prefixes=CHECK,O0
; RUN: llc -fast-isel -fast-isel-abort=1 -mtriple=x86_64-unknown-unknown -stop-after=finalize-isel %s -o - | \
; RUN:    FileCheck %s --check-prefixes=CHECK,IREF

; A debug use of the loaded value must not prevent folding the load, and the
; debug value must not refer to the now undefined vreg. At -O0 this is a
; DBG_VALUE; with instruction referencing it is a DBG_INSTR_REF that becomes an
; undef DBG_VALUE_LIST.

; CHECK-LABEL: name: fold_load_dbg
; CHECK-NOT:   MOV64rm
; O0:          DBG_VALUE $noreg, $noreg,
; IREF:        DBG_VALUE_LIST {{.*}}, $noreg,
; CHECK-NEXT:  ADD64rm
define i64 @fold_load_dbg(ptr %a, i64 %b) !dbg !4 {
  %1 = load i64, ptr %a, align 8
    #dbg_value(i64 %1, !6, !DIExpression(), !7)
  %2 = add i64 %1, %b
  ret i64 %2
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !DISubroutineType(types: !{})
!4 = distinct !DISubprogram(name: "fold_load_dbg", scope: !1, file: !1, type: !3, spFlags: DISPFlagDefinition, unit: !0)
!5 = !DIBasicType(name: "long", size: 64, encoding: DW_ATE_signed)
!6 = !DILocalVariable(name: "x", scope: !4, file: !1, type: !5)
!7 = !DILocation(line: 1, scope: !4)
