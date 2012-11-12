; RUN: llc < %s -mtriple=i686-linux-pc -mcpu=corei7 | FileCheck %s

define <2 x float> @foo(i32 %x, i32 %y, <2 x float> %v) {
  %t1 = uitofp i32 %x to float
  %t2 = insertelement <2 x float> undef, float %t1, i32 0
  %t3 = uitofp i32 %y to float
  %t4 = insertelement <2 x float> %t2, float %t3, i32 1
  %t5 = fmul <2 x float> %v, %t4
  ret <2 x float> %t5
; CHECK: foo
; CHECK: or
; CHECK: subpd
; CHECK: cvtpd2ps
; CHECK: ret
}

define <2 x float> @bar(<2 x i32> %in) {
  %r = uitofp <2 x i32> %in to <2 x float>
  ret <2 x float> %r
; CHECK: bar
; CHECK: or
; CHECK: subpd
; CHECK: cvtpd2ps
; CHECK: ret
}
