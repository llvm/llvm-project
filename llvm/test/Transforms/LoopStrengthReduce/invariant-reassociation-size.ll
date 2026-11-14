; RUN: opt -passes=loop-reduce -lsr-max-invariant-reassociation-size=4 -verify-each -S %s | FileCheck %s --check-prefixes=CHECK,KEEP
; RUN: opt -passes=loop-reduce -lsr-max-invariant-reassociation-size=5 -verify-each -S %s | FileCheck %s --check-prefixes=CHECK,SPLIT
; RUN: opt -passes=loop-reduce -lsr-max-invariant-reassociation-size=256 -verify-each -S %s | FileCheck %s --check-prefixes=CHECK,SPLIT
; RUN: opt -passes=loop-reduce -lsr-max-invariant-reassociation-size=0 -verify-each -S %s | FileCheck %s --check-prefix=DISABLED
; REQUIRES: x86-registered-target
;
; A positive threshold protects only expressions larger than the limit.
; Zero disables the limit. Keeping the invariant sum together must still expose
; the stride-eight index. The invariant address arithmetic stays outside the loop.
;
; CHECK-LABEL: define void @whole_base(
; CHECK: entry:
; CHECK: [[BASE:%.*]] = getelementptr i8, ptr %p, i64
; CHECK: loop:
; CHECK-NEXT: %i = phi i64
; CHECK-NEXT: %scaled = shl i64 %i, 3
; CHECK-NEXT: {{%.*}} = getelementptr i8, ptr [[BASE]], i64 %scaled
; CHECK-NEXT: store volatile i8 1,
; CHECK-NEXT: %next = add nuw i64 %i, 1
; CHECK-NEXT: %more = icmp ult i64 %next, %n
; CHECK-NEXT: br i1 %more,
;
target triple = "x86_64-unknown-linux-gnu"

define void @whole_base(ptr %p, i64 %n, i64 %a, i64 %b, i64 %c, i64 %d, i64 %e, i64 %f, i64 %g, i64 %h) {
entry:
  %ab = add i64 %a, %b
  %abc = add i64 %ab, %c
  %abcd = add i64 %abc, %d
  %abcde = add i64 %abcd, %e
  %abcdef = add i64 %abcde, %f
  %abcdefg = add i64 %abcdef, %g
  %base = add i64 %abcdefg, %h
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %loop ]
  %scaled = shl i64 %i, 3
  %offset = add i64 %base, %scaled
  %address = getelementptr i8, ptr %p, i64 %offset
  store volatile i8 1, ptr %address
  %next = add nuw i64 %i, 1
  %more = icmp ult i64 %next, %n
  br i1 %more, label %loop, label %exit
exit:
  ret void
}

; The invariant base p + 4*len has five SCEV nodes. A limit equal to its
; size permits reassociation; a smaller limit keeps it intact, introducing a
; separate pointer IV. Disabling the limit also permits the original split.
;
; KEEP-LABEL: define i32 @small_base(
; KEEP: loop:
; KEEP-NEXT: {{%.*}} = phi ptr
; KEEP-NEXT: %iv = phi i64
; SPLIT-LABEL: define i32 @small_base(
; SPLIT: loop:
; SPLIT-NEXT: %iv = phi i64
; SPLIT-NEXT: %cond = icmp eq i64 %iv, 0
; DISABLED-LABEL: define i32 @small_base(
; DISABLED: loop:
; DISABLED-NEXT: %iv = phi i64
; DISABLED-NEXT: %cond = icmp eq i64 %iv, 0
;
define i32 @small_base(ptr %p, i64 %len, i32 %x) {
entry:
  br label %loop
loop:
  %iv = phi i64 [ %next, %backedge ], [ %len, %entry ]
  %next = add nsw i64 %iv, -1
  %cond = icmp eq i64 %iv, 0
  br i1 %cond, label %exit, label %backedge
backedge:
  %addr = getelementptr inbounds i32, ptr %p, i64 %next
  %loaded = load atomic i32, ptr %addr unordered, align 4
  %equal = icmp eq i32 %loaded, %x
  br i1 %equal, label %failure, label %loop
exit:
  ret i32 -1
failure:
  unreachable
}
