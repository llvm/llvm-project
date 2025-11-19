; RUN: opt -disable-output -passes='print<access-info><allow-partial>,print<access-info>' %s 2>&1 | FileCheck %s --check-prefixes=ALLOW-BEFORE
; RUN: opt -disable-output -passes='print<access-info>,print<access-info><allow-partial>' %s 2>&1 | FileCheck %s --check-prefixes=ALLOW-AFTER

; Check that we get the right results when loop access analysis is run twice,
; once without partial results and once with.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128-Fn32"

define void @gep_loaded_offset(ptr %p, ptr %q, ptr %r, i32 %n) {
; ALLOW-BEFORE-LABEL: 'gep_loaded_offset'
; ALLOW-BEFORE-NEXT:    while.body:
; ALLOW-BEFORE-NEXT:      Report: cannot identify array bounds
; ALLOW-BEFORE-NEXT:      Dependences:
; ALLOW-BEFORE-NEXT:      Run-time memory checks:
; ALLOW-BEFORE-NEXT:      Check 0:
; ALLOW-BEFORE-NEXT:        Comparing group GRP0:
; ALLOW-BEFORE-NEXT:          %p.addr = phi ptr [ %incdec.ptr, %while.body ], [ %p, %entry ]
; ALLOW-BEFORE-NEXT:        Against group GRP1:
; ALLOW-BEFORE-NEXT:        ptr %r
; ALLOW-BEFORE-NEXT:      Grouped accesses:
; ALLOW-BEFORE-NEXT:        Group GRP0:
; ALLOW-BEFORE-NEXT:          (Low: %p High: (4 + (4 * (zext i32 (-1 + %n)<nsw> to i64))<nuw><nsw> + %p))
; ALLOW-BEFORE-NEXT:            Member: {%p,+,4}<nuw><%while.body>
; ALLOW-BEFORE-NEXT:        Group GRP1:
; ALLOW-BEFORE-NEXT:          (Low: %r High: (8 + %r))
; ALLOW-BEFORE-NEXT:            Member: %r
; ALLOW-BEFORE-NEXT:      Generated run-time checks are incomplete
; ALLOW-BEFORE-EMPTY:
; ALLOW-BEFORE-NEXT:      Non vectorizable stores to invariant address were not found in loop.
; ALLOW-BEFORE-NEXT:      SCEV assumptions:
; ALLOW-BEFORE-EMPTY:
; ALLOW-BEFORE-NEXT:      Expressions re-written:
;
; ALLOW-BEFORE-LABEL: 'gep_loaded_offset'
; ALLOW-BEFORE-NEXT:    while.body:
; ALLOW-BEFORE-NEXT:      Report: cannot identify array bounds
; ALLOW-BEFORE-NEXT:      Dependences:
; ALLOW-BEFORE-NEXT:      Run-time memory checks:
; ALLOW-BEFORE-NEXT:      Grouped accesses:
; ALLOW-BEFORE-EMPTY:
; ALLOW-BEFORE-NEXT:      Non vectorizable stores to invariant address were not found in loop.
; ALLOW-BEFORE-NEXT:      SCEV assumptions:
; ALLOW-BEFORE-EMPTY:
; ALLOW-BEFORE-NEXT:      Expressions re-written:
;
; ALLOW-AFTER-LABEL: 'gep_loaded_offset'
; ALLOW-AFTER-NEXT:    while.body:
; ALLOW-AFTER-NEXT:      Report: cannot identify array bounds
; ALLOW-AFTER-NEXT:      Dependences:
; ALLOW-AFTER-NEXT:      Run-time memory checks:
; ALLOW-AFTER-NEXT:      Grouped accesses:
; ALLOW-AFTER-EMPTY:
; ALLOW-AFTER-NEXT:      Non vectorizable stores to invariant address were not found in loop.
; ALLOW-AFTER-NEXT:      SCEV assumptions:
; ALLOW-AFTER-EMPTY:
; ALLOW-AFTER-NEXT:      Expressions re-written:
;
; ALLOW-AFTER-LABEL: 'gep_loaded_offset'
; ALLOW-AFTER-NEXT:    while.body:
; ALLOW-AFTER-NEXT:      Report: cannot identify array bounds
; ALLOW-AFTER-NEXT:      Dependences:
; ALLOW-AFTER-NEXT:      Run-time memory checks:
; ALLOW-AFTER-NEXT:      Check 0:
; ALLOW-AFTER-NEXT:        Comparing group GRP0:
; ALLOW-AFTER-NEXT:          %p.addr = phi ptr [ %incdec.ptr, %while.body ], [ %p, %entry ]
; ALLOW-AFTER-NEXT:        Against group GRP1:
; ALLOW-AFTER-NEXT:        ptr %r
; ALLOW-AFTER-NEXT:      Grouped accesses:
; ALLOW-AFTER-NEXT:        Group GRP0:
; ALLOW-AFTER-NEXT:          (Low: %p High: (4 + (4 * (zext i32 (-1 + %n)<nsw> to i64))<nuw><nsw> + %p))
; ALLOW-AFTER-NEXT:            Member: {%p,+,4}<nuw><%while.body>
; ALLOW-AFTER-NEXT:        Group GRP1:
; ALLOW-AFTER-NEXT:          (Low: %r High: (8 + %r))
; ALLOW-AFTER-NEXT:            Member: %r
; ALLOW-AFTER-NEXT:      Generated run-time checks are incomplete
; ALLOW-AFTER-EMPTY:
; ALLOW-AFTER-NEXT:      Non vectorizable stores to invariant address were not found in loop.
; ALLOW-AFTER-NEXT:      SCEV assumptions:
; ALLOW-AFTER-EMPTY:
; ALLOW-AFTER-NEXT:      Expressions re-written:
;
entry:
  br label %while.body

while.body:
  %n.addr = phi i32 [ %dec, %while.body ], [ %n, %entry ]
  %p.addr = phi ptr [ %incdec.ptr, %while.body ], [ %p, %entry ]
  %dec = add nsw i32 %n.addr, -1
  %rval = load i64, ptr %r, align 4
  %arrayidx = getelementptr inbounds i32, ptr %q, i64 %rval
  %val = load i32, ptr %arrayidx, align 4
  %incdec.ptr = getelementptr inbounds nuw i8, ptr %p.addr, i64 4
  store i32 %val, ptr %p.addr, align 4
  %tobool.not = icmp eq i32 %dec, 0
  br i1 %tobool.not, label %while.end, label %while.body

while.end:
  ret void
}
