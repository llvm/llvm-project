; RUN: llc -mcpu=pwr9 -mtriple=powerpc64le-unknown-linux-gnu < %s \
; RUN:   -stop-before=finalize-isel | \
; RUN:   FileCheck -check-prefix=BEFORE-FINALIZE-ISEL %s
; RUN: llc -mcpu=pwr9 -mtriple=powerpc64le-unknown-linux-gnu < %s \
; RUN:   -stop-after=finalize-isel | \
; RUN:   FileCheck -check-prefix=AFTER-FINALIZE-ISEL %s
; RUN: llc -mcpu=pwr9 -mtriple=powerpc64le-unknown-linux-gnu < %s \
; RUN:   -verify-machineinstrs | FileCheck %s
; RUN: llc -mcpu=pwr7 -mtriple=powerpc64le-unknown-linux-gnu < %s \
; RUN:   -verify-machineinstrs | FileCheck -check-prefix=CHECK-PWR7 %s

define double @test_setrndi() {
entry:
  %0 = tail call double @llvm.ppc.setrnd(i32 2)
  ret double %0

; BEFORE-FINALIZE-ISEL: test_setrndi
; BEFORE-FINALIZE-ISEL: SETRNDi 2, implicit-def dead $rm, implicit $rm

; AFTER-FINALIZE-ISEL:  test_setrndi
; AFTER-FINALIZE-ISEL:  MFFS implicit $rm
; AFTER-FINALIZE-ISEL:  MTFSB0 31, implicit-def $rm
; AFTER-FINALIZE-ISEL:  MTFSB1 30, implicit-def $rm

; CHECK-LABEL: @test_setrndi
; CHECK:      # %bb.0:
; CHECK-DAG:   mffs 1 
; CHECK-DAG:   mtfsb0 31
; CHECK-DAG:   mtfsb1 30
; CHECK:       blr
}

define double @test_setrnd(i32 signext %x) {
entry:
  %0 = tail call double @llvm.ppc.setrnd(i32 %x)
  ret double %0

; BEFORE-FINALIZE-ISEL: test_setrnd
; BEFORE-FINALIZE-ISEL: SETRND %1, implicit-def dead $rm, implicit $rm

; AFTER-FINALIZE-ISEL: test_setrnd
; AFTER-FINALIZE-ISEL: MFFS implicit $rm
; AFTER-FINALIZE-ISEL: MTFSF 255, %7, 0, 0, implicit-def $rm

; CHECK-LABEL: @test_setrnd
; CHECK:      # %bb.0:
; CHECK-DAG:   mffs 1 
; CHECK-DAG:   mffprd [[REG1:[0-9]+]], 1 
; CHECK-DAG:   rldimi [[REG1]], 3, 0, 62
; CHECK-DAG:   mtfprd [[REG2:[0-9]+]], [[REG1]]
; CHECK-DAG:   mtfsf 255, [[REG2]] 
; CHECK:       blr

; CHECK-PWR7-LABEL: @test_setrnd
; CHECK-PWR7:      # %bb.0:
; CHECK-PWR7-DAG:   mffs 1 
; CHECK-PWR7-DAG:   stfd 1, -8(1) 
; CHECK-PWR7-DAG:   ld [[REG1:[0-9]+]], -8(1) 
; CHECK-PWR7-DAG:   rldimi [[REG1]], 3, 0, 62
; CHECK-PWR7-DAG:   std [[REG1]], -16(1)
; CHECK-PWR7-DAG:   lfd [[REG2:[0-9]+]], -16(1)
; CHECK-PWR7-DAG:   mtfsf 255, [[REG2]] 
; CHECK-PWR7:       blr
}

declare double @llvm.ppc.setrnd(i32) 

; A debug-only use of the old FPSCR value must not make us read the FPSCR.
define void @test_setrndi_debug_use() !dbg !3 {
entry:
  %0 = tail call double @llvm.ppc.setrnd(i32 2)
    #dbg_value(double %0, !5, !DIExpression(), !6)
  ret void

; AFTER-FINALIZE-ISEL:      name: test_setrndi_debug_use
; AFTER-FINALIZE-ISEL:      [[OLD:%[0-9]+]]:f8rc = IMPLICIT_DEF
; AFTER-FINALIZE-ISEL-NEXT: MTFSB0 31, implicit-def $rm
; AFTER-FINALIZE-ISEL-NEXT: MTFSB1 30, implicit-def $rm
; AFTER-FINALIZE-ISEL-NEXT: DBG_VALUE $noreg

; CHECK-LABEL: @test_setrndi_debug_use
; CHECK:       prologue_end
; CHECK-NEXT:  mtfsb0 31
; CHECK-NEXT:  mtfsb1 30
; CHECK-NEXT:  #DEBUG_VALUE: test_setrndi_debug_use:old <- undef
; CHECK-NEXT:  blr
}

define void @test_setflm_debug_use(double %x) !dbg !7 {
entry:
  %0 = tail call double @llvm.ppc.setflm(double %x)
    #dbg_value(double %0, !8, !DIExpression(), !9)
  ret void

; AFTER-FINALIZE-ISEL:      name: test_setflm_debug_use
; AFTER-FINALIZE-ISEL:      [[OLD:%[0-9]+]]:f8rc = IMPLICIT_DEF
; AFTER-FINALIZE-ISEL-NEXT: MTFSF 255, %{{[0-9]+}}, 0, 0, implicit-def $rm
; AFTER-FINALIZE-ISEL-NEXT: DBG_VALUE $noreg

; CHECK-LABEL: @test_setflm_debug_use
; CHECK:       prologue_end
; CHECK-NEXT:  mtfsf 255, 1
; CHECK-NEXT:  #DEBUG_VALUE: test_setflm_debug_use:old <- undef
; CHECK-NEXT:  blr
}

declare double @llvm.ppc.setflm(double)

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2}

!0 = distinct !DICompileUnit(language: DW_LANG_C11, file: !1, emissionKind: FullDebug)
!1 = !DIFile(filename: "t.c", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = distinct !DISubprogram(name: "test_setrndi_debug_use", scope: !1, file: !1, type: !4, spFlags: DISPFlagDefinition, unit: !0)
!4 = !DISubroutineType(types: !{null})
!5 = !DILocalVariable(name: "old", scope: !3)
!6 = !DILocation(line: 0, scope: !3)
!7 = distinct !DISubprogram(name: "test_setflm_debug_use", scope: !1, file: !1, type: !4, spFlags: DISPFlagDefinition, unit: !0)
!8 = !DILocalVariable(name: "old", scope: !7)
!9 = !DILocation(line: 0, scope: !7)
