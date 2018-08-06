; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=-fpcvt < %s | FileCheck %s
; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=-fpcvt -ppc-gen-isel=false < %s | FileCheck %s --check-prefix=CHECK-NO-ISEL
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define float @test(i64 %x) nounwind readnone {
entry:
  %conv = sitofp i64 %x to float
  ret float %conv
}

; Verify that we get the code sequence needed to avoid double-rounding.
; Note that only parts of the sequence are checked for here, to allow
; for minor code generation differences.

;CHECK-LABEL: test
;CHECK-NO-ISEL-LABEL: test
; CHECK: sradi [[REG1:[0-9]+]], 3, 53
; CHECK: addi [[REG2:[0-9]+]], [[REG1]], 1
; CHECK: cmpldi [[REG2]], 1
; CHECK: isel [[REG3:[0-9]+]], {{[0-9]+}}, 3, 1
; CHECK-NO-ISEL: rldicr [[REG2:[0-9]+]], {{[0-9]+}}, 0, 52
; CHECK-NO-ISEL: bc 12, 1, [[TRUE:.LBB[0-9]+]]
; CHECK-NO-ISEL: b [[SUCCESSOR:.LBB[0-9]+]]
; CHECK-NO-ISEL-NEXT: [[TRUE]]
; CHECK-NO-ISEL-NEXT: addi {{[0-9]+}}, [[REG2]], 0
; CHECK-NO-ISEL-NEXT: [[SUCCESSOR]]
; CHECK-NO-ISEL: std {{[0-9]+}}, -{{[0-9]+}}(1)
; CHECK: std [[REG3]], -{{[0-9]+}}(1)


; Also check that with -enable-unsafe-fp-math we do not get that extra
; code sequence.  Simply verify that there is no "isel" present.

; RUN: llc -verify-machineinstrs -mcpu=pwr7 -mattr=-fpcvt -enable-unsafe-fp-math < %s | FileCheck %s -check-prefix=CHECK-UNSAFE
; CHECK-UNSAFE-NOT: isel

