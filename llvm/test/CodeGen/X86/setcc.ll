; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s
; rdar://7329206

; Use sbb x, x to materialize carry bit in a GPR. The value is either
; all 1's or all 0's.

define zeroext i16 @t1(i16 zeroext %x) nounwind readnone ssp {
entry:
; CHECK: t1:
; CHECK: seta %al
; CHECK: movzbl %al, %eax
; CHECK: shll $5, %eax
  %0 = icmp ugt i16 %x, 26                        ; <i1> [#uses=1]
  %iftmp.1.0 = select i1 %0, i16 32, i16 0        ; <i16> [#uses=1]
  ret i16 %iftmp.1.0
}

define zeroext i16 @t2(i16 zeroext %x) nounwind readnone ssp {
entry:
; CHECK: t2:
; CHECK: sbbl %eax, %eax
; CHECK: andl $32, %eax
  %0 = icmp ult i16 %x, 26                        ; <i1> [#uses=1]
  %iftmp.0.0 = select i1 %0, i16 32, i16 0        ; <i16> [#uses=1]
  ret i16 %iftmp.0.0
}

define i64 @t3(i64 %x) nounwind readnone ssp {
entry:
; CHECK: t3:
; CHECK: sbbq %rax, %rax
; CHECK: andq $64, %rax
  %0 = icmp ult i64 %x, 18                        ; <i1> [#uses=1]
  %iftmp.2.0 = select i1 %0, i64 64, i64 0        ; <i64> [#uses=1]
  ret i64 %iftmp.2.0
}
