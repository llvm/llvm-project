; Verify BOTH pipeline placements of DeadBranchElimination are required.
; Each function below is only optimizable at one of the two placements, so
; removing either placement makes one of the positive checks fail:
;
;   @circular_limit needs the BEFORE-INLINE placement (in module
;   simplification, ahead of GlobalCleanupPM's SimplifyCFG). Its dead body
;   `limit += 1` is side-effect free, so a later SimplifyCFG speculates it
;   into a select (`limit += zext(b == limit)`), destroying the branch
;   before a post-inline pass instance could ever see it.
;
;   @fill needs the AFTER-INLINE placement (start of module optimization).
;   Before inlining, the dead `len == cap` branch lives in @push where
;   nothing is provable about its arguments, and @fill contains no
;   candidate branch at all; only after @push is inlined does the
;   circular-dependency pattern exist in a single function. Its dead body
;   contains a call, so SimplifyCFG can never speculate it away and it
;   survives to the later placement.
;
; RUN: opt -passes='default<O2>' -S %s | FileCheck %s
; RUN: opt -passes='default<O2>' -enable-dead-branch-elim=false -S %s \
; RUN:   | FileCheck %s --check-prefix=DISABLED

; int circular_limit() {
;   int a = 0, b = 0, limit = 100;
;   while (a < limit) {
;     if (b == limit)   // unreachable, but modifies limit
;       limit += 1;
;     a++; b++;
;   }
;   return b;
; }
; Raw (un-optimized) clang output shape: locals in allocas so the branch
; reaches the before-inline placement exactly as it would from clang.

; CHECK-LABEL: @circular_limit(
; CHECK-NOT:     br
; CHECK:         ret i32 100
;
; DISABLED-LABEL: @circular_limit(
; DISABLED:         br i1

define i32 @circular_limit() nounwind {
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  %limit = alloca i32, align 4
  store i32 0, ptr %a, align 4
  store i32 0, ptr %b, align 4
  store i32 100, ptr %limit, align 4
  br label %header

header:
  %a.cur = load i32, ptr %a, align 4
  %limit.cur = load i32, ptr %limit, align 4
  %in.bounds = icmp slt i32 %a.cur, %limit.cur
  br i1 %in.bounds, label %body, label %exit

body:
  %b.cur = load i32, ptr %b, align 4
  %limit.cur2 = load i32, ptr %limit, align 4
  %hit.limit = icmp eq i32 %b.cur, %limit.cur2
  br i1 %hit.limit, label %bump, label %latch

bump:
  %limit.cur3 = load i32, ptr %limit, align 4
  %limit.next = add nsw i32 %limit.cur3, 1
  store i32 %limit.next, ptr %limit, align 4
  br label %latch

latch:
  %a.cur2 = load i32, ptr %a, align 4
  %a.next = add nsw i32 %a.cur2, 1
  store i32 %a.next, ptr %a, align 4
  %b.cur2 = load i32, ptr %b, align 4
  %b.next = add nsw i32 %b.cur2, 1
  store i32 %b.next, ptr %b, align 4
  br label %header, !llvm.loop !0

exit:
  %b.final = load i32, ptr %b, align 4
  ret i32 %b.final
}

; Vec-push shape: @fill(n) reserves capacity n, then pushes n times; the
; grow path inside @push is dead but only provably so after inlining.

; CHECK-LABEL: @fill(
; CHECK-NOT:     @grow
; CHECK:         ret i64 %n
;
; DISABLED-LABEL: @fill(
; DISABLED:         call {{.*}}@grow

declare i64 @grow(i64)

define internal i64 @push(i64 %len, i64 %cap) nounwind {
entry:
  %full = icmp eq i64 %len, %cap
  br i1 %full, label %do.grow, label %done

do.grow:
  %cap.grown = call i64 @grow(i64 %cap)
  br label %done

done:
  %cap.out = phi i64 [ %cap, %entry ], [ %cap.grown, %do.grow ]
  ret i64 %cap.out
}

define i64 @fill(i64 %n) nounwind {
entry:
  br label %header

header:
  %i = phi i64 [ 0, %entry ], [ %i.next, %latch ]
  %cap = phi i64 [ %n, %entry ], [ %cap.next, %latch ]
  %more = icmp ult i64 %i, %n
  br i1 %more, label %latch, label %exit

latch:
  %cap.next = call i64 @push(i64 %i, i64 %cap)
  %i.next = add nuw i64 %i, 1
  br label %header, !llvm.loop !0

exit:
  ret i64 %cap
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.mustprogress"}
