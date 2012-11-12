; RUN: opt -O3 -S -analyze -scalar-evolution %s | FileCheck %s
;
; This file contains phase ordering tests for scalar evolution.
; Test that the standard passes don't obfuscate the IR so scalar evolution can't
; recognize expressions.

; CHECK: test1
; The loop body contains two increments by %div.
; Make sure that 2*%div is recognizable, and not expressed as a bit mask of %d.
; CHECK: -->  {%p,+,(2 * (%d /u 4) * sizeof(i32))}
define void @test1(i64 %d, i32* %p) nounwind uwtable ssp {
entry:
  %div = udiv i64 %d, 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %p.addr.0 = phi i32* [ %p, %entry ], [ %add.ptr1, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp ne i32 %i.0, 64
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  store i32 0, i32* %p.addr.0, align 4
  %add.ptr = getelementptr inbounds i32* %p.addr.0, i64 %div
  store i32 1, i32* %add.ptr, align 4
  %add.ptr1 = getelementptr inbounds i32* %add.ptr, i64 %div
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; CHECK: test1a
; Same thing as test1, but it is even more tempting to fold 2 * (%d /u 2)
; CHECK: -->  {%p,+,(2 * (%d /u 2) * sizeof(i32))}
define void @test1a(i64 %d, i32* %p) nounwind uwtable ssp {
entry:
  %div = udiv i64 %d, 2
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %p.addr.0 = phi i32* [ %p, %entry ], [ %add.ptr1, %for.inc ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp ne i32 %i.0, 64
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  store i32 0, i32* %p.addr.0, align 4
  %add.ptr = getelementptr inbounds i32* %p.addr.0, i64 %div
  store i32 1, i32* %add.ptr, align 4
  %add.ptr1 = getelementptr inbounds i32* %add.ptr, i64 %div
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %inc = add i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}
