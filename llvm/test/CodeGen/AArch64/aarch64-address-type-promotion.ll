; RUN: llc < %s -o - | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-n32:64"
target triple = "arm64-apple-macosx10.9"

; Check that sexts get promoted above adds.
define void @foo(ptr nocapture %a, i32 %i) {
entry:
; CHECK-LABEL: _foo:
; CHECK: add
; CHECK-NEXT: ldp
; CHECK-NEXT: add
; CHECK-NEXT: str
; CHECK-NEXT: ret
  %add = add nsw i32 %i, 1
  %idxprom = sext i32 %add to i64
  %arrayidx = getelementptr inbounds i32, ptr %a, i64 %idxprom
  %0 = load i32, ptr %arrayidx, align 4
  %add1 = add nsw i32 %i, 2
  %idxprom2 = sext i32 %add1 to i64
  %arrayidx3 = getelementptr inbounds i32, ptr %a, i64 %idxprom2
  %1 = load i32, ptr %arrayidx3, align 4
  %add4 = add nsw i32 %1, %0
  %idxprom5 = sext i32 %i to i64
  %arrayidx6 = getelementptr inbounds i32, ptr %a, i64 %idxprom5
  store i32 %add4, ptr %arrayidx6, align 4
  ret void
}
