; RUN: opt < %s -S -ipsccp | FileCheck %s

; After the first round of Solver.Solve(), the return value of @testf still
; undefined as we hit a branch on undef. Therefore the conditional branch on
; @testf's return value in @bar is unknown. In ResolvedUndefsIn, we force the
; false branch to be feasible. We later discover that @testf actually
; returns true, so we end up with an unfolded "br i1 true".
define void @test1() {
; CHECK-LABEL: @test1(
; CHECK-LABEL: if.then:
; CHECK:         [[CALL:%.+]] = call i1 @testf()
; CHECK-NEXT:    br i1 true, label %if.end, label %if.then
;
entry:
  br label %if.then
if.then:                                          ; preds = %entry, %if.then
  %foo = phi i32 [ 0, %entry], [ %next, %if.then]
  %next = add i32 %foo, 1
  %call = call i1 @testf()
  br i1 %call, label %if.end, label %if.then

if.end:                                           ; preds = %if.then, %entry
  ret void
}

define internal i1 @testf() {
; CHECK-LABEL: define internal i1 @testf(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[IF_END3:%.*]]
; CHECK:       if.end3:
; CHECK-NEXT:    ret i1 undef
;
entry:
  br i1 undef, label %if.then1, label %if.end3

if.then1:                                         ; preds = %if.end
  br label %if.end3

if.end3:                                          ; preds = %if.then1, %entry
  ret i1 true
}


; Call sites in unreachable blocks should not be a problem.
; CHECK-LABEL: define i1 @test2() {
; CHECK-NEXT: entry:
; CHECK-NEXT:   br label %if.end
; CHECK-LABEL: if.end:                                           ; preds = %entry
; CHECK-NEXT:   %call2 = call i1 @testf()
; CHECK-NEXT:   ret i1 true
define i1 @test2() {
entry:
  br label %if.end

if.then:                                          ; preds = %entry, %if.then
  %call = call i1 @testf()
  br i1 %call, label %if.end, label %if.then

if.end:                                           ; preds = %if.then, %entry
  %call2 = call i1 @testf()
  ret i1 %call2
}
