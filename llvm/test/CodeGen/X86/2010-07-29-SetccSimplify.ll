; RUN: llc < %s -mtriple=x86_64-apple-darwin | FileCheck %s

define i32 @extend2bit_v2(i32 %val) {
entry:
  %0 = trunc i32 %val to i2                       ; <i2> [#uses=1]
  %1 = sext i2 %0 to i32                          ; <i32> [#uses=1]
  %2 = icmp eq i32 %1, 3                          ; <i1> [#uses=1]
  %3 = zext i1 %2 to i32                          ; <i32> [#uses=1]
  ret i32 %3
}

; CHECK: extend2bit_v2:
; CHECK: xorl	%eax, %eax
; CHECK-NEXT: ret
