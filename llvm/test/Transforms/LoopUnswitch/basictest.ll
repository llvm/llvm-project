; RUN: opt < %s -loop-unswitch -verify-loop-info -S < %s 2>&1 | FileCheck %s

define i32 @test(i32* %A, i1 %C) {
entry:
	br label %no_exit
no_exit:		; preds = %no_exit.backedge, %entry
	%i.0.0 = phi i32 [ 0, %entry ], [ %i.0.0.be, %no_exit.backedge ]		; <i32> [#uses=3]
	%gep.upgrd.1 = zext i32 %i.0.0 to i64		; <i64> [#uses=1]
	%tmp.7 = getelementptr i32, i32* %A, i64 %gep.upgrd.1		; <i32*> [#uses=4]
	%tmp.13 = load i32, i32* %tmp.7		; <i32> [#uses=2]
	%tmp.14 = add i32 %tmp.13, 1		; <i32> [#uses=1]
	store i32 %tmp.14, i32* %tmp.7
	br i1 %C, label %then, label %endif
then:		; preds = %no_exit
	%tmp.29 = load i32, i32* %tmp.7		; <i32> [#uses=1]
	%tmp.30 = add i32 %tmp.29, 2		; <i32> [#uses=1]
	store i32 %tmp.30, i32* %tmp.7
	%inc9 = add i32 %i.0.0, 1		; <i32> [#uses=2]
	%tmp.112 = icmp ult i32 %inc9, 100000		; <i1> [#uses=1]
	br i1 %tmp.112, label %no_exit.backedge, label %return
no_exit.backedge:		; preds = %endif, %then
	%i.0.0.be = phi i32 [ %inc9, %then ], [ %inc, %endif ]		; <i32> [#uses=1]
	br label %no_exit
endif:		; preds = %no_exit
	%inc = add i32 %i.0.0, 1		; <i32> [#uses=2]
	%tmp.1 = icmp ult i32 %inc, 100000		; <i1> [#uses=1]
	br i1 %tmp.1, label %no_exit.backedge, label %return
return:		; preds = %endif, %then
	ret i32 %tmp.13
}

; This simple test would normally unswitch, but should be inhibited by the presence of
; the noduplicate call.

; CHECK-LABEL: @test2(
define i32 @test2(i32* %var) {
  %mem = alloca i32
  store i32 2, i32* %mem
  %c = load i32, i32* %mem

  br label %loop_begin

loop_begin:

  %var_val = load i32, i32* %var

  switch i32 %c, label %default [
      i32 1, label %inc
      i32 2, label %dec
  ]

inc:
  call void @incf() noreturn nounwind
  br label %loop_begin
dec:
; CHECK: call void @decf()
; CHECK-NOT: call void @decf()
  call void @decf() noreturn nounwind noduplicate
  br label %loop_begin
default:
  br label %loop_exit
loop_exit:
  ret i32 0
; CHECK: }
}

; This simple test would normally unswitch, but should be inhibited by the presence of
; the convergent call that is not control-dependent on the unswitch condition.

; CHECK-LABEL: @test3(
define i32 @test3(i32* %var) {
  %mem = alloca i32
  store i32 2, i32* %mem
  %c = load i32, i32* %mem

  br label %loop_begin

loop_begin:

  %var_val = load i32, i32* %var

; CHECK: call void @conv()
; CHECK-NOT: call void @conv()
  call void @conv() convergent

  switch i32 %c, label %default [
      i32 1, label %inc
      i32 2, label %dec
  ]

inc:
  call void @incf() noreturn nounwind
  br label %loop_begin
dec:
  call void @decf() noreturn nounwind
  br label %loop_begin
default:
  br label %loop_exit
loop_exit:
  ret i32 0
; CHECK: }
}

; Make sure we unswitch %a == 0 out of the loop.
;
; CHECK: define void @and_i2_as_switch_input(i2
; CHECK: entry:
; This is an indication that the loop has been unswitched.
; CHECK: icmp eq i2 %a, 0
; CHECK: br
; There should be no more unswitching after the 1st unswitch.
; CHECK-NOT: icmp eq
; CHECK: ret
define void @and_i2_as_switch_input(i2 %a) {
entry:
  br label %for.body

for.body:
  %i = phi i2 [ 0, %entry ], [ %inc, %for.inc ]
  %and = and i2 %a, %i
  %and1 = and i2 %and, %i
  switch i2 %and1, label %sw.default [
    i2 0, label %sw.bb
    i2 1, label %sw.bb1
  ]

sw.bb:
  br label %sw.epilog

sw.bb1:
  br label %sw.epilog

sw.default:
  br label %sw.epilog

sw.epilog:
  br label %for.inc

for.inc:
  %inc = add nsw i2 %i, 1
  %cmp = icmp slt i2 %inc, 3 
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

; Make sure we unswitch %a == !0 out of the loop.
;
; CHECK: define void @or_i2_as_switch_input(i2
; CHECK: entry:
; This is an indication that the loop has been unswitched.
; CHECK: icmp eq i2 %a, -1
; CHECK: br
; There should be no more unswitching after the 1st unswitch.
; CHECK-NOT: icmp eq
; CHECK: ret
define void @or_i2_as_switch_input(i2 %a) {
entry:
  br label %for.body

for.body:
  %i = phi i2 [ 0, %entry ], [ %inc, %for.inc ]
  %or = or i2 %a, %i
  %or1 = or i2 %or, %i
  switch i2 %or1, label %sw.default [
    i2 2, label %sw.bb
    i2 3, label %sw.bb1
  ]

sw.bb:
  br label %sw.epilog

sw.bb1:
  br label %sw.epilog

sw.default:
  br label %sw.epilog

sw.epilog:
  br label %for.inc

for.inc:
  %inc = add nsw i2 %i, 1
  %cmp = icmp slt i2 %inc, 3 
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

; Make sure we unswitch %a == !0 out of the loop. Even we do not
; have it as a case value. Unswitching it out allows us to simplify
; the or operator chain.
;
; CHECK: define void @or_i2_as_switch_input_unswitch_default(i2
; CHECK: entry:
; This is an indication that the loop has been unswitched.
; CHECK: icmp eq i2 %a, -1
; CHECK: br
; There should be no more unswitching after the 1st unswitch.
; CHECK-NOT: icmp eq
; CHECK: ret
define void @or_i2_as_switch_input_unswitch_default(i2 %a) {
entry:
  br label %for.body

for.body:
  %i = phi i2 [ 0, %entry ], [ %inc, %for.inc ]
  %or = or i2 %a, %i
  %or1 = or i2 %or, %i
  switch i2 %or1, label %sw.default [
    i2 1, label %sw.bb
    i2 2, label %sw.bb1
  ]

sw.bb:
  br label %sw.epilog

sw.bb1:
  br label %sw.epilog

sw.default:
  br label %sw.epilog

sw.epilog:
  br label %for.inc

for.inc:
  %inc = add nsw i2 %i, 1
  %cmp = icmp slt i2 %inc, 3 
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

; Make sure we don't unswitch, as we can not find an input value %a
; that will effectively unswitch 0 or 3 out of the loop.
;
; CHECK: define void @and_or_i2_as_switch_input(i2
; CHECK: entry:
; This is an indication that the loop has NOT been unswitched.
; CHECK-NOT: icmp
; CHECK: br
define void @and_or_i2_as_switch_input(i2 %a) {
entry:
  br label %for.body

for.body:
  %i = phi i2 [ 0, %entry ], [ %inc, %for.inc ]
  %and = and i2 %a, %i 
  %or = or i2 %and, %i
  switch i2 %or, label %sw.default [
    i2 0, label %sw.bb
    i2 3, label %sw.bb1
  ]

sw.bb:
  br label %sw.epilog

sw.bb1:
  br label %sw.epilog

sw.default:
  br label %sw.epilog

sw.epilog:
  br label %for.inc

for.inc:
  %inc = add nsw i2 %i, 1
  %cmp = icmp slt i2 %inc, 3 
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

; Make sure we don't unswitch, as we can not find an input value %a
; that will effectively unswitch true/false out of the loop.
;
; CHECK: define void @and_or_i1_as_branch_input(i1
; CHECK: entry:
; This is an indication that the loop has NOT been unswitched.
; CHECK-NOT: icmp
; CHECK: br
define void @and_or_i1_as_branch_input(i1 %a) {
entry:
  br label %for.body

for.body:
  %i = phi i1 [ 0, %entry ], [ %inc, %for.inc ]
  %and = and i1 %a, %i 
  %or = or i1 %and, %i
  br i1 %or, label %sw.bb, label %sw.bb1

sw.bb:
  br label %sw.epilog

sw.bb1:
  br label %sw.epilog

sw.epilog:
  br label %for.inc

for.inc:
  %inc = add nsw i1 %i, 1
  %cmp = icmp slt i1 %inc, 1 
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}

declare void @incf() noreturn
declare void @decf() noreturn
declare void @conv() convergent
