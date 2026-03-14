; REQUIRES: asserts

; RUN: opt -S -passes=loop-fusion -loop-fusion-dependence-analysis=da -debug-only=loop-fusion -disable-output < %s 2>&1 | FileCheck %s --check-prefix=CHECK-DA
; RUN: opt -S -passes=loop-fusion -loop-fusion-dependence-analysis=scev -debug-only=loop-fusion -disable-output < %s 2>&1 | FileCheck %s --check-prefix=CHECK-SCEV

define void @loop_invariant(i32 %N) {
; CHECK-DA: Performing Loop Fusion on function loop_invariant
; CHECK-DA: Safe to fuse due to a loop-invariant non-anti dependency
; CHECK-SCEV: Performing Loop Fusion on function loop_invariant
; CHECK-SCEV: Fusion done
;
pre1:
  %ptr = alloca i32, align 4
  br label %body1

body1:  ; preds = %pre1, %body1
  %i = phi i32 [%i_next, %body1], [0, %pre1]
  %i_next = add i32 1, %i
  %cond = icmp ne i32 %i, %N
  store i32 3, ptr %ptr
  br i1 %cond, label %body1, label %pre2

pre2:
  br label %body2

body2:  ; preds = %pre2, %body2
  %i2 = phi i32 [%i_next2, %body2], [0, %pre2]
  %i_next2 = add i32 1, %i2
  %cond2 = icmp ne i32 %i2, %N
  store i32 3, ptr %ptr
  br i1 %cond2, label %body2, label %exit

exit:
  ret void
}

; TODO: improve SCEV check to detect the loop-invariant anti dependence with
; scalar access and prevent fusion.
define void @anti_loop_invariant(i32 %N) {
; CHECK-DA: Performing Loop Fusion on function anti_loop_invariant
; CHECK-DA: Memory dependencies do not allow fusion!
; CHECK-SCEV: Performing Loop Fusion on function anti_loop_invariant
; XFAIL-CHECK-SCEV: Memory dependencies do not allow fusion!
;
pre1:
  %ptr = alloca i32, align 4
  store i32 1, ptr %ptr
  br label %body1

body1:  ; preds = %pre1, %body1
  %i = phi i32 [%i_next, %body1], [0, %pre1]
  %i_next = add i32 1, %i
  %cond = icmp ne i32 %i, %N
  %v = load i32, ptr %ptr
  br i1 %cond, label %body1, label %pre2

pre2:
  br label %body2

body2:  ; preds = %pre2, %body2
  %i2 = phi i32 [%i_next2, %body2], [0, %pre2]
  %i_next2 = add i32 1, %i2
  %cond2 = icmp ne i32 %i2, %N
  store i32 3, ptr %ptr
  br i1 %cond2, label %body2, label %exit

exit:
  ret void
}
