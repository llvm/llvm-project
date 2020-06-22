; RUN: llc -mcpu=pwr9 -mtriple=powerpc64le-unknown-linux-gnu < %s \
; RUN:   -stop-before=finalize-isel -verify-machineinstrs | \
; RUN:   FileCheck -check-prefix=BEFORE-FINALIZE-ISEL %s
; RUN: llc -mcpu=pwr9 -mtriple=powerpc64le-unknown-linux-gnu < %s \
; RUN:   -stop-after=finalize-isel -verify-machineinstrs | \
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
; AFTER-FINALIZE-ISEL:  MTFSB0 31, implicit-def $rm, implicit $rm
; AFTER-FINALIZE-ISEL:  MTFSB1 30, implicit-def $rm, implicit $rm

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
; BEFORE-FINALIZE-ISEL: SETRND killed %1, implicit-def dead $rm, implicit $rm

; AFTER-FINALIZE-ISEL: test_setrnd
; AFTER-FINALIZE-ISEL: MFFS implicit $rm
; AFTER-FINALIZE-ISEL: MTFSF 255, %7, 0, 0

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

