; RUN: opt -passes='loop(simple-loop-unswitch),verify<loops>' -S < %s | FileCheck %s
; RUN: opt -verify-memoryssa -passes='loop-mssa(simple-loop-unswitch),verify<loops>' -S < %s | FileCheck %s

declare void @unknown()
declare void @unknown2()

@y = global i64 0, align 8

; The following is approximately:
; void f(bool *x) {
;   for (int i = 0; i < 1; ++i) {
;     if (*x) {
;       if (y)
;         unknown();
;       else
;         break;
;     }
;   }
; }
; With MemorySanitizer, the loop can not be unswitched on "y", because "y" could
; be uninitialized when x == false.
; Test that the branch on "y" is inside the loop (after the first unconditional
; branch).

define void @may_not_execute_trivial(i1* %x) sanitize_memory {
; CHECK-LABEL: @may_not_execute_trivial(
entry:
  %y = load i64, i64* @y, align 8
  %y.cmp = icmp eq i64 %y, 0
  br label %for.body
; CHECK: %[[Y:.*]] = load i64, i64* @y
; CHECK: %[[YCMP:.*]] = icmp eq i64 %[[Y]], 0
; CHECK-NOT: br i1
; CHECK: br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %x.load = load i1, i1* %x
  br i1 %x.load, label %for.inc, label %if.then
; CHECK: %[[XLOAD:.*]] = load i1, i1* %x
; CHECK: br i1 %[[XLOAD]]

if.then:
  br i1 %y.cmp, label %for.end, label %if.then4
; CHECK: br i1 %[[YCMP]]

if.then4:
  call void @unknown()
  br label %for.inc

for.inc:
  %inc = add nsw i32 %i, 1
  %cmp = icmp slt i32 %inc, 1
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}


; The same as above, but "y" is a function parameter instead of a global.
; This shows that it is not enough to suppress hoisting of load instructions,
; the actual problem is in the speculative branching.

define void @may_not_execute2_trivial(i1* %x, i1 %y) sanitize_memory {
; CHECK-LABEL: @may_not_execute2_trivial(
entry:
  br label %for.body
; CHECK-NOT: br i1
; CHECK: br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %x.load = load i1, i1* %x
  br i1 %x.load, label %for.inc, label %if.then
; CHECK: %[[XLOAD:.*]] = load i1, i1* %x
; CHECK: br i1 %[[XLOAD]]

if.then:
  br i1 %y, label %for.end, label %if.then4
; CHECK: br i1 %y

if.then4:
  call void @unknown()
  br label %for.inc

for.inc:
  %inc = add nsw i32 %i, 1
  %cmp = icmp slt i32 %inc, 1
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
}


; The following is approximately:
; void f() {
;   for (int i = 0; i < 1; ++i) {
;     if (y)
;       unknown();
;     else
;       break;
;   }
; }
; "if (y)" is guaranteed to execute; the loop can be unswitched.

define void @must_execute_trivial() sanitize_memory {
; CHECK-LABEL: @must_execute_trivial(
entry:
  %y = load i64, i64* @y, align 8
  %y.cmp = icmp eq i64 %y, 0
  br label %for.body
; CHECK:   %[[Y:.*]] = load i64, i64* @y
; CHECK:   %[[YCMP:.*]] = icmp eq i64 %[[Y]], 0
; CHECK:   br i1 %[[YCMP]], label %[[EXIT_SPLIT:.*]], label %[[PH:.*]]
;
; CHECK: [[PH]]:
; CHECK:   br label %for.body

for.body:
  %i = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  br i1 %y.cmp, label %for.end, label %if.then4
; CHECK: br label %if.then4

if.then4:
  call void @unknown()
  br label %for.inc

for.inc:
  %inc = add nsw i32 %i, 1
  %cmp = icmp slt i32 %inc, 1
  br i1 %cmp, label %for.body, label %for.end

for.end:
  ret void
; CHECK: for.end:
; CHECK:   br label %[[EXIT_SPLIT]]
;
; CHECK: [[EXIT_SPLIT]]:
; CHECK:   ret void
}
