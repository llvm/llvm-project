; RUN: llc -mcpu=pwr9 -mtriple=powerpc64le-unknown-linux-gnu < %s \
; RUN:   -verify-machineinstrs | FileCheck %s
; RUN: llc -mcpu=pwr7 -mtriple=powerpc64le-unknown-linux-gnu < %s \
; RUN:   -verify-machineinstrs | FileCheck -check-prefix=CHECK-PWR7 %s

define double @test_setrndi() {
entry:
  %0 = tail call double @llvm.ppc.setrnd(i32 2)
  ret double %0

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

; CHECK-LABEL: @test_setrnd
; CHECK:      # %bb.0:
; CHECK-DAG:   mffs 1 
; CHECK-DAG:   mffprd [[REG1:[0-9]+]], 1 
; CHECK-DAG:   rldimi [[REG1]], 3, 0, 62
; CHECK-DAG:   mtvsrd [[REG2:[0-9]+]], [[REG1]]
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

