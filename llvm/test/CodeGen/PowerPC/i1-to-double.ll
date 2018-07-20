; RUN: llc -verify-machineinstrs -mcpu=ppc32 -mtriple=powerpc-unknown-linux-gnu < %s | FileCheck %s
define double @test(i1 %X) {
        %Y = uitofp i1 %X to double
        ret double %Y
}

; CHECK-LABEL: @test
; CHECK-DAG: addis 3, 4, .LCPI
; CHECK-DAG: addis 4, 4, .LCPI
; CHECK-DAG: andi. {{[0-9]+}}, 3, 1
; CHECK-NEXT: bc 12, 1, [[TRUE:.LBB[0-9]+]]
; CHECK-NEXT: b [[SUCCESSOR:.LBB[0-9]+]]
; CHECK-NEXT: [[TRUE]]
; CHECK-NEXT: addi 3, 4, 0
; CHECK-NEXT: [[SUCCESSOR]]
; CHECK-NEXT: lfs 1, 0(3)
; CHECK-NEXT: blr
