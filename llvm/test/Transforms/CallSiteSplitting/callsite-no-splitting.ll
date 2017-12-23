; RUN: opt < %s -callsite-splitting -S | FileCheck %s
; RUN: opt < %s  -passes='function(callsite-splitting)' -S | FileCheck %s

define i32 @callee(i32*, i32, i32) {
  ret i32 10
}

; CHECK-LABEL: @test_preds_equal
; CHECK-NOT: split
; CHECK: br i1 %cmp, label %Tail, label %Tail
define i32 @test_preds_equal(i32* %a, i32 %v, i32 %p) {
TBB:
  %cmp = icmp eq i32* %a, null
  br i1 %cmp, label %Tail, label %Tail
Tail:
  %r = call i32 @callee(i32* %a, i32 %v, i32 %p)
  ret i32 %r
}
