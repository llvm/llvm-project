; RUN: llc -mcpu=pwr7 < %s | FileCheck %s
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

; CHECK: sradi [[REGISTER:[0-9]+]], 3, 53
; CHECK: addi [[REGISTER:[0-9]+]], [[REGISTER]], 1
; CHECK: cmpldi 0, [[REGISTER]], 1
; CHECK: isel [[REGISTER:[0-9]+]], {{[0-9]+}}, 3, 1
; CHECK: std [[REGISTER]], -{{[0-9]+}}(1)


; Also check that with -enable-unsafe-fp-math we do not get that extra
; code sequence.  Simply verify that there is no "isel" present.

; RUN: llc -mcpu=pwr7 -enable-unsafe-fp-math < %s | FileCheck %s -check-prefix=UNSAFE
; CHECK-UNSAFE-NOT: isel

