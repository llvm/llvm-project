; RUN: llc < %s -mtriple=armv4-unknown-eabi | FileCheck %s -check-prefix=V4
; RUN: llc < %s -mtriple=armv5-unknown-eabi | FileCheck %s
; RUN: llc < %s -mtriple=armv6-unknown-eabi | FileCheck %s

define i32 @bar(i32 %a) nounwind {
entry:
  %0 = tail call i32 @foo(i32 %a) nounwind ; <i32> [#uses=1]
  %1 = add nsw i32 %0, 3                          ; <i32> [#uses=1]
; CHECK: pop {r11, pc}
; V4: pop
; V4-NEXT: mov pc, lr
  ret i32 %1
}

declare i32 @foo(i32)
