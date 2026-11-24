; RUN: opt -mtriple=amdgcn-amd-amdhsa -passes=loop-unroll -disable-output < %s

; Fully unrolling the first loop creates shared SCEVs for the second loop's
; trip count. Checking their uniformity should avoid revisiting shared nodes,
; which would take exponential time.

define void @shared_scev_trip_count(ptr %out, i32 inreg %n) {
entry:
  br label %powers

powers:
  %j = phi i32 [ 0, %entry ], [ %j.next, %powers ]
  %value = phi i32 [ %n, %entry ], [ %square, %powers ]
  %square = mul i32 %value, %value
  %j.next = add i32 %j, 1
  %finished = icmp eq i32 %j.next, 40
  br i1 %finished, label %preheader, label %powers, !llvm.loop !0

preheader:
  br label %loop

loop:
  %iv = phi i32 [ 0, %preheader ], [ %next, %loop ]
  store volatile i32 %iv, ptr %out
  %next = add i32 %iv, 1
  %done = icmp uge i32 %next, %square
  br i1 %done, label %exit, label %loop

exit:
  ret void
}

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.full"}
