; RUN: opt < %s -passes=loop-predication -S | FileCheck %s
; RUN: opt -S -passes='require<scalar-evolution>,loop-mssa(loop-predication)' -verify-memoryssa < %s 2>&1 | FileCheck %s

;; This is a simplified copy of @unswitch_exit_form test that should trigger loop-predication
;; activity and properly bail out when discovering that widenable check does not lead to deopt.
;;
;; Error checking is rather silly here - it should pass compilation successfully,
;; in bad case it will just timeout.
;;
define i32 @unswitch_exit_form_with_endless_loop(ptr %array, i32 %length, i32 %n, i1 %cond_0) {
; CHECK-LABEL: @unswitch_exit_form_with_endless_loop
entry:
  %widenable_cond = call i1 @llvm.experimental.widenable.condition()
  %exiplicit_guard_cond = and i1 %cond_0, %widenable_cond
  br i1 %exiplicit_guard_cond, label %loop.preheader, label %not_really_a_deopt, !prof !0

not_really_a_deopt:
  br label %looping

looping:
  ;; synthetic corner case that demonstrates the need for more careful traversal
  ;; of unique successors when walking through the exit for profitability checks.
  br label %looping

loop.preheader:
  br label %loop

loop:
  %loop.acc = phi i32 [ %loop.acc.next, %guarded ], [ 0, %loop.preheader ]
  %i = phi i32 [ %i.next, %guarded ], [ 0, %loop.preheader ]
  %within.bounds = icmp ult i32 %i, %length
  br i1 %within.bounds, label %guarded, label %not_really_a_deopt, !prof !0

guarded:
  %i.i64 = zext i32 %i to i64
  %array.i.ptr = getelementptr inbounds i32, ptr %array, i64 %i.i64
  %array.i = load i32, ptr %array.i.ptr, align 4
  store i32 0, ptr %array.i.ptr
  %loop.acc.next = add i32 %loop.acc, %array.i
  %i.next = add nuw i32 %i, 1
  %continue = icmp ult i32 %i.next, %n
  br i1 %continue, label %loop, label %exit

exit:
  %result = phi i32 [ %loop.acc.next, %guarded ]
  ret i32 %result
}

declare void @unknown()

declare i1 @llvm.experimental.widenable.condition()
declare i32 @llvm.experimental.deoptimize.i32(...)

!0 = !{!"branch_weights", i32 1048576, i32 1}
!1 = !{i32 1, i32 -2147483648}
!2 = !{i32 0, i32 50}
