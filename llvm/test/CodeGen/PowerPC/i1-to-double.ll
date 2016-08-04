; RUN: llc -verify-machineinstrs -march=ppc32 -mcpu=ppc32 -mtriple=powerpc-unknown-linux-gnu < %s | FileCheck %s
define double @test(i1 %X) {
        %Y = uitofp i1 %X to double
        ret double %Y
}

; CHECK-LABEL: @test

; CHECK: andi. {{[0-9]+}}, 3, 1
; CHECK: bc 12, 1,

; CHECK: li 3, .LCP[[L1:[A-Z0-9_]+]]@l
; CHECK: addis 3, 3, .LCP[[L1]]@ha
; CHECK: lfs 1, 0(3)
; CHECK: blr

; CHECK: li 3, .LCP[[L2:[A-Z0-9_]+]]@l
; CHECK: addis 3, 3, .LCP[[L2]]@ha
; CHECK: lfs 1, 0(3)
; CHECK: blr

