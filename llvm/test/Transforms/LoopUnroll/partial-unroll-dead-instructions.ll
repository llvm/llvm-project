; RUN: opt -S < %s -passes=loop-unroll -unroll-allow-partial=1 | FileCheck %s
;
; Bugpointed test that triggered UB while cleaning up dead
; instructions after simplifying indvars

; We just check that some unrolling happened here - the assert we've
; added to ValueHandleBase::operator* would fire if the bug was still
; present.
; CHECK: atomicrmw volatile add i32*
; CHECK: atomicrmw volatile add i32*
; CHECK: atomicrmw volatile add i32*

@global = external global i32, align 4

define void @widget() {
bb:
  br label %bb1

bb1:
  br label %bb2

bb2:
  %tmp = phi i32 [ 0, %bb1 ], [ %tmp34, %bb33 ]
  %tmp3 = phi i32 [ 0, %bb1 ], [ %tmp34, %bb33 ]
  %tmp26 = and i32 %tmp, 1073741823
  %tmp27 = getelementptr inbounds i32, i32* @global, i32 %tmp26
  %tmp28 = atomicrmw volatile add i32* %tmp27, i32 1 monotonic
  %tmp29 = icmp ugt i32 %tmp28, 23
  %tmp30 = shl i32 %tmp, 6
  %tmp31 = add i32 %tmp30, undef
  %tmp32 = add i32 %tmp31, %tmp28
  store i32 undef, i32* undef, align 4
  br label %bb33

bb33:
  %tmp34 = add nuw nsw i32 %tmp3, 1
  %tmp35 = icmp ult i32 %tmp3, 15
  br i1 %tmp35, label %bb2, label %bb36

bb36:
  br label %bb1
}
