; RUN: opt < %s -passes=callsite-splitting -S | FileCheck %s
; RUN: opt < %s -passes='function(callsite-splitting)' -S | FileCheck %s

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

define void @fn1(i16 %p1) {
entry:
  ret void
}

define void @fn2() {
  ret void

; Unreachable code below

for.inc:                                          ; preds = %for.inc
  br i1 undef, label %for.end6, label %for.inc

for.end6:                                         ; preds = %for.inc
  br i1 undef, label %lor.rhs, label %lor.end

lor.rhs:                                          ; preds = %for.end6
  br label %lor.end

lor.end:                                          ; preds = %for.end6, %lor.rhs
  call void @fn1(i16 0)
  ret void
}
