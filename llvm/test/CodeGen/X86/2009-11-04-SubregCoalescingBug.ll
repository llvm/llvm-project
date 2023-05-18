; RUN: llc < %s -mtriple=x86_64-apple-darwin11 | FileCheck %s
; rdar://7362871

define void @bar(i32 %b, i32 %a) nounwind optsize ssp {
entry:
; CHECK:     leal 15(%rsi), %edi
; CHECK-NOT: movl
; CHECK:     _foo
  %0 = add i32 %a, 15                             ; <i32> [#uses=1]
  %1 = zext i32 %0 to i64                         ; <i64> [#uses=1]
  tail call void @foo(i64 %1) nounwind
  ret void
}

declare void @foo(i64)
