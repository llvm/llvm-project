; RUN: opt < %s -loop-unswitch -enable-new-pm=0 -verify-loop-info -S < %s 2>&1 | FileCheck %s
; RUN: opt < %s -loop-unswitch -enable-new-pm=0 -verify-loop-info -enable-mssa-loop-dependency=true -verify-memoryssa -S < %s 2>&1 | FileCheck %s

@sink = global i32 0, align 4
@y = global i64 0, align 8

; The following is approximately:
; void f(bool x, int p, int q) {
;   volatile bool x2 = x;
;   for (int i = 0; i < 1; ++i) {
;     if (x2) {
;       if (y)
;         sink = p;
;       else
;         sink = q;
;     }
;   }
; }
; With MemorySanitizer, the loop can not be unswitched on "y", because "y" could
; be uninitialized when x == false.
; Test that the branch on "y" is inside the loop (after the first unconditional
; branch).

define void @may_not_execute(i1 zeroext %x, i32 %p, i32 %q) sanitize_memory {
; CHECK-LABEL: @may_not_execute(
entry:
; CHECK: %[[Y:.*]] = load i64, i64* @y, align 8
; CHECK: %[[YB:.*]] = icmp eq i64 %[[Y]], 0
; CHECK-NOT: br i1
; CHECK: br label
; CHECK: br i1 %[[YB]]

  %x2 = alloca i8, align 1
  %frombool1 = zext i1 %x to i8
  store volatile i8 %frombool1, i8* %x2, align 1
  %0 = load i64, i64* @y, align 8
  %tobool3 = icmp eq i64 %0, 0
  br label %for.body

for.body:
  %i.01 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %x2.0. = load volatile i8, i8* %x2, align 1
  %tobool2 = icmp eq i8 %x2.0., 0
  br i1 %tobool2, label %for.inc, label %if.then

if.then:
  br i1 %tobool3, label %if.else, label %if.then4

if.then4:
  store volatile i32 %p, i32* @sink, align 4
  br label %for.inc

if.else:
  store volatile i32 %q, i32* @sink, align 4
  br label %for.inc

for.inc:
  %inc = add nsw i32 %i.01, 1
  %cmp = icmp slt i32 %inc, 1
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}


; The same as above, but "y" is a function parameter instead of a global.
; This shows that it is not enough to suppress hoisting of load instructions,
; the actual problem is in the speculative branching.

define void @may_not_execute2(i1 zeroext %x, i1 zeroext %y, i32 %p, i32 %q) sanitize_memory {
; CHECK-LABEL: @may_not_execute2(
entry:
; CHECK-NOT: br i1
; CHECK: br label
; CHECK: br i1 %y,
  %x2 = alloca i8, align 1
  %frombool2 = zext i1 %x to i8
  store volatile i8 %frombool2, i8* %x2, align 1
  br label %for.body

for.body:
  %i.01 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %x2.0. = load volatile i8, i8* %x2, align 1
  %tobool3 = icmp eq i8 %x2.0., 0
  br i1 %tobool3, label %for.inc, label %if.then

if.then:
  br i1 %y, label %if.then5, label %if.else

if.then5:
  store volatile i32 %p, i32* @sink, align 4
  br label %for.inc

if.else:
  store volatile i32 %q, i32* @sink, align 4
  br label %for.inc

for.inc:
  %inc = add nsw i32 %i.01, 1
  %cmp = icmp slt i32 %inc, 1
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}


; The following is approximately:
; void f(bool x, int p, int q) {
;   volatile bool x2 = x;
;   for (int i = 0; i < 1; ++i) {
;     if (y)
;       sink = p;
;     else
;       sink = q;
;   }
; }
; "if (y)" is guaranteed to execute; the loop can be unswitched.

define void @must_execute(i1 zeroext %x, i32 %p, i32 %q) sanitize_memory {
; CHECK-LABEL: @must_execute(
entry:
; CHECK:       %[[Y:.*]] = load i64, i64* @y, align 8
; CHECK-NEXT:  %[[YB:.*]] = icmp eq i64 %[[Y]], 0
; CHECK-NEXT:  br i1 %[[YB]],

  %x2 = alloca i8, align 1
  %frombool1 = zext i1 %x to i8
  store volatile i8 %frombool1, i8* %x2, align 1
  %0 = load i64, i64* @y, align 8
  %tobool2 = icmp eq i64 %0, 0
  br label %for.body

for.body:
  %i.01 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  br i1 %tobool2, label %if.else, label %if.then

if.then:
  store volatile i32 %p, i32* @sink, align 4
  br label %for.inc

if.else:
  store volatile i32 %q, i32* @sink, align 4
  br label %for.inc

for.inc:
  %inc = add nsw i32 %i.01, 1
  %cmp = icmp slt i32 %inc, 1
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}
