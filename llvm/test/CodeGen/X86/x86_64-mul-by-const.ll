; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s
; Formerly there were two shifts.  rdar://8771012.

define i32 @f9188_mul365384439_shift27(i32 %A) nounwind {
; CHECK:  imulq $365384439,
; CHECK:  shrq  $59, %rax
        %tmp1 = udiv i32 %A, 1577682821         ; <i32> [#uses=1]
        ret i32 %tmp1
}
