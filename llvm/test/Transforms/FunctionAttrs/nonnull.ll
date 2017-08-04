; RUN: opt -S -functionattrs -enable-nonnull-arg-prop %s | FileCheck %s
declare nonnull i8* @ret_nonnull()

; Return a pointer trivially nonnull (call return attribute)
define i8* @test1() {
; CHECK: define nonnull i8* @test1
  %ret = call i8* @ret_nonnull()
  ret i8* %ret
}

; Return a pointer trivially nonnull (argument attribute)
define i8* @test2(i8* nonnull %p) {
; CHECK: define nonnull i8* @test2
  ret i8* %p
}

; Given an SCC where one of the functions can not be marked nonnull,
; can we still mark the other one which is trivially nonnull
define i8* @scc_binder() {
; CHECK: define i8* @scc_binder
  call i8* @test3()
  ret i8* null
}

define i8* @test3() {
; CHECK: define nonnull i8* @test3
  call i8* @scc_binder()
  %ret = call i8* @ret_nonnull()
  ret i8* %ret
}

; Given a mutual recursive set of functions, we can mark them
; nonnull if neither can ever return null.  (In this case, they
; just never return period.)
define i8* @test4_helper() {
; CHECK: define noalias nonnull i8* @test4_helper
  %ret = call i8* @test4()
  ret i8* %ret
}

define i8* @test4() {
; CHECK: define noalias nonnull i8* @test4
  %ret = call i8* @test4_helper()
  ret i8* %ret
}

; Given a mutual recursive set of functions which *can* return null
; make sure we haven't marked them as nonnull.
define i8* @test5_helper() {
; CHECK: define noalias i8* @test5_helper
  %ret = call i8* @test5()
  ret i8* null
}

define i8* @test5() {
; CHECK: define noalias i8* @test5
  %ret = call i8* @test5_helper()
  ret i8* %ret
}

; Local analysis, but going through a self recursive phi
define i8* @test6() {
entry:
; CHECK: define nonnull i8* @test6
  %ret = call i8* @ret_nonnull()
  br label %loop
loop:
  %phi = phi i8* [%ret, %entry], [%phi, %loop]
  br i1 undef, label %loop, label %exit
exit:
  ret i8* %phi
}

; Test propagation of nonnull callsite args back to caller.

declare void @use1(i8* %x)
declare void @use2(i8* %x, i8* %y);
declare void @use3(i8* %x, i8* %y, i8* %z);

declare void @use1nonnull(i8* nonnull %x);
declare void @use2nonnull(i8* nonnull %x, i8* nonnull %y);
declare void @use3nonnull(i8* nonnull %x, i8* nonnull %y, i8* nonnull %z);

declare i8 @use1safecall(i8* %x) readonly nounwind ; readonly+nounwind guarantees that execution continues to successor

; Can't extend non-null to parent for any argument because the 2nd call is not guaranteed to execute.

define void @parent1(i8* %a, i8* %b, i8* %c) {
; CHECK-LABEL: @parent1(i8* %a, i8* %b, i8* %c)
; CHECK-NEXT:    call void @use3(i8* %c, i8* %a, i8* %b)
; CHECK-NEXT:    call void @use3nonnull(i8* %b, i8* %c, i8* %a)
; CHECK-NEXT:    ret void
;
  call void @use3(i8* %c, i8* %a, i8* %b)
  call void @use3nonnull(i8* %b, i8* %c, i8* %a)
  ret void
}

; Extend non-null to parent for all arguments.

define void @parent2(i8* %a, i8* %b, i8* %c) {
; CHECK-LABEL: @parent2(i8* nonnull %a, i8* nonnull %b, i8* nonnull %c)
; CHECK-NEXT:    call void @use3nonnull(i8* %b, i8* %c, i8* %a)
; CHECK-NEXT:    call void @use3(i8* %c, i8* %a, i8* %b)
; CHECK-NEXT:    ret void
;
  call void @use3nonnull(i8* %b, i8* %c, i8* %a)
  call void @use3(i8* %c, i8* %a, i8* %b)
  ret void
}

; Extend non-null to parent for 1st argument.

define void @parent3(i8* %a, i8* %b, i8* %c) {
; CHECK-LABEL: @parent3(i8* nonnull %a, i8* %b, i8* %c)
; CHECK-NEXT:    call void @use1nonnull(i8* %a)
; CHECK-NEXT:    call void @use3(i8* %c, i8* %b, i8* %a)
; CHECK-NEXT:    ret void
;
  call void @use1nonnull(i8* %a)
  call void @use3(i8* %c, i8* %b, i8* %a)
  ret void
}

; Extend non-null to parent for last 2 arguments.

define void @parent4(i8* %a, i8* %b, i8* %c) {
; CHECK-LABEL: @parent4(i8* %a, i8* nonnull %b, i8* nonnull %c)
; CHECK-NEXT:    call void @use2nonnull(i8* %c, i8* %b)
; CHECK-NEXT:    call void @use2(i8* %a, i8* %c)
; CHECK-NEXT:    call void @use1(i8* %b)
; CHECK-NEXT:    ret void
;
  call void @use2nonnull(i8* %c, i8* %b)
  call void @use2(i8* %a, i8* %c)
  call void @use1(i8* %b)
  ret void
}

; The callsite must execute in order for the attribute to transfer to the parent.
; It appears benign to extend non-null to the parent in this case, but we can't do that
; because it would incorrectly propagate the wrong information to its callers.

define void @parent5(i8* %a, i1 %a_is_notnull) {
; CHECK-LABEL: @parent5(i8* %a, i1 %a_is_notnull)
; CHECK-NEXT:    br i1 %a_is_notnull, label %t, label %f
; CHECK:       t:
; CHECK-NEXT:    call void @use1nonnull(i8* %a)
; CHECK-NEXT:    ret void
; CHECK:       f:
; CHECK-NEXT:    ret void
;
  br i1 %a_is_notnull, label %t, label %f
t:
  call void @use1nonnull(i8* %a)
  ret void
f:
  ret void
}

; The callsite must execute in order for the attribute to transfer to the parent.
; The volatile load might trap, so there's no guarantee that we'll ever get to the call.

define i8 @parent6(i8* %a, i8* %b) {
; CHECK-LABEL: @parent6(i8* %a, i8* %b)
; CHECK-NEXT:    [[C:%.*]] = load volatile i8, i8* %b
; CHECK-NEXT:    call void @use1nonnull(i8* %a)
; CHECK-NEXT:    ret i8 [[C]]
;
  %c = load volatile i8, i8* %b
  call void @use1nonnull(i8* %a)
  ret i8 %c
}

; The nonnull callsite is guaranteed to execute, so the argument must be nonnull throughout the parent.

define i8 @parent7(i8* %a) {
; CHECK-LABEL: @parent7(i8* nonnull %a)
; CHECK-NEXT:    [[RET:%.*]] = call i8 @use1safecall(i8* %a)
; CHECK-NEXT:    call void @use1nonnull(i8* %a)
; CHECK-NEXT:    ret i8 [[RET]]
;
  %ret = call i8 @use1safecall(i8* %a)
  call void @use1nonnull(i8* %a)
  ret i8 %ret
}

; Make sure that an invoke works similarly to a call.

declare i32 @esfp(...)

define i1 @parent8(i8* %a, i8* %bogus1, i8* %b) personality i8* bitcast (i32 (...)* @esfp to i8*){
; CHECK-LABEL: @parent8(i8* nonnull %a, i8* nocapture readnone %bogus1, i8* nonnull %b)
; CHECK-NEXT:  entry:
; CHECK-NEXT:    invoke void @use2nonnull(i8* %a, i8* %b)
; CHECK-NEXT:    to label %cont unwind label %exc
; CHECK:       cont:
; CHECK-NEXT:    [[NULL_CHECK:%.*]] = icmp eq i8* %b, null
; CHECK-NEXT:    ret i1 [[NULL_CHECK]]
; CHECK:       exc:
; CHECK-NEXT:    [[LP:%.*]] = landingpad { i8*, i32 }
; CHECK-NEXT:    filter [0 x i8*] zeroinitializer
; CHECK-NEXT:    unreachable
;
entry:
  invoke void @use2nonnull(i8* %a, i8* %b)
  to label %cont unwind label %exc

cont:
  %null_check = icmp eq i8* %b, null
  ret i1 %null_check

exc:
  %lp = landingpad { i8*, i32 }
  filter [0 x i8*] zeroinitializer
  unreachable
}

