; RUN: opt -passes='default<O3>,print<scalar-evolution>' -S < %s 2>&1 | FileCheck %s

target datalayout = "e-m:m-p:40:64:64:32-i32:32-i16:16-i8:8-n32"

;
; This file contains phase ordering tests for scalar evolution.
; Test that the standard passes don't obfuscate the IR so scalar evolution can't
; recognize expressions.

; CHECK: test1
; The loop body contains two increments by %div.
; Make sure that 2*%div is recognizable, and not expressed as a bit mask of %d.
; CHECK: -->  {%p,+,(8 * (%d /u 4))}
define void @test1(i32 %d, ptr %p) nounwind uwtable ssp {
entry:
  %div = udiv i32 %d, 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %p.addr.0 = phi ptr [ %p, %entry ], [ %add.ptr1, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp ne i32 %i.0, 64
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  store i32 0, ptr %p.addr.0, align 4
  %add.ptr = getelementptr inbounds i32, ptr %p.addr.0, i32 %div
  store i32 1, ptr %add.ptr, align 4
  %add.ptr1 = getelementptr inbounds i32, ptr %add.ptr, i32 %div
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; CHECK: test1a
; Same thing as test1, but it is even more tempting to fold 2 * (%d /u 2)
; CHECK: -->  {%p,+,(8 * (%d /u 2))}
define void @test1a(i32 %d, ptr %p) nounwind uwtable ssp {
entry:
  %div = udiv i32 %d, 2
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %p.addr.0 = phi ptr [ %p, %entry ], [ %add.ptr1, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp ne i32 %i.0, 64
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  store i32 0, ptr %p.addr.0, align 4
  %add.ptr = getelementptr inbounds i32, ptr %p.addr.0, i32 %div
  store i32 1, ptr %add.ptr, align 4
  %add.ptr1 = getelementptr inbounds i32, ptr %add.ptr, i32 %div
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

@array = weak global [101 x i32] zeroinitializer, align 32		; <ptr> [#uses=1]

; CHECK: Loop %bb: backedge-taken count is 100

define void @test_range_ref1a(i32 %x) {
entry:
	br label %bb

bb:		; preds = %bb, %entry
	%i.01.0 = phi i32 [ 100, %entry ], [ %tmp4, %bb ]		; <i32> [#uses=2]
	%tmp1 = getelementptr [101 x i32], ptr @array, i32 0, i32 %i.01.0		; <ptr> [#uses=1]
	store i32 %x, ptr %tmp1
	%tmp4 = add i32 %i.01.0, -1		; <i32> [#uses=2]
	%tmp7 = icmp sgt i32 %tmp4, -1		; <i1> [#uses=1]
	br i1 %tmp7, label %bb, label %return

return:		; preds = %bb
	ret void
}

define i32 @test_loop_idiom_recogize(i32 %x, i32 %y, ptr %lam, ptr %alp) nounwind {
bb1.thread:
	br label %bb1

bb1:		; preds = %bb1, %bb1.thread
	%indvar = phi i32 [ 0, %bb1.thread ], [ %indvar.next, %bb1 ]		; <i32> [#uses=4]
	%i.0.reg2mem.0 = sub i32 255, %indvar		; <i32> [#uses=2]
	%0 = getelementptr i32, ptr %alp, i32 %i.0.reg2mem.0		; <ptr> [#uses=1]
	%1 = load i32, ptr %0, align 4		; <i32> [#uses=1]
	%2 = getelementptr i32, ptr %lam, i32 %i.0.reg2mem.0		; <ptr> [#uses=1]
	store i32 %1, ptr %2, align 4
	%3 = sub i32 254, %indvar		; <i32> [#uses=1]
	%4 = icmp slt i32 %3, 0		; <i1> [#uses=1]
	%indvar.next = add i32 %indvar, 1		; <i32> [#uses=1]
	br i1 %4, label %bb2, label %bb1

bb2:		; preds = %bb1
	%tmp10 = mul i32 %indvar, %x		; <i32> [#uses=1]
	%z.0.reg2mem.0 = add i32 %tmp10, %y		; <i32> [#uses=1]
	%5 = add i32 %z.0.reg2mem.0, %x		; <i32> [#uses=1]
	ret i32 %5
}

declare void @use(i1)

declare void @llvm.experimental.guard(i1, ...)

; This tests getRangeRef acts as intended with different idx size.
; CHECK: Loop %loop: Unpredictable constant max backedge-taken count.
define void @test_range_ref1(i8 %t) {
 entry:
  %t.ptr = inttoptr i8 %t to ptr
  %p.42 = inttoptr i8 42 to ptr
  %cmp1 = icmp slt ptr %t.ptr, %p.42
  call void(i1, ...) @llvm.experimental.guard(i1 %cmp1) [ "deopt"() ]
  br label %loop

 loop:
  %idx = phi ptr [ %t.ptr, %entry ], [ %snext, %loop ]
  %snext = getelementptr inbounds i8, ptr %idx, i64 1
  %c = icmp slt ptr %idx, %p.42
  call void @use(i1 %c)
  %be = icmp slt ptr %snext, %p.42
  br i1 %be, label %loop, label %exit

 exit:
  ret void
}

