; RUN: opt -S -loop-sink < %s | FileCheck %s
; RUN: opt -S -passes=loop-sink < %s | FileCheck %s

@g = global i32 0, align 4

;     b1
;    /  \
;   b2  b6
;  /  \  |
; b3  b4 |
;  \  /  |
;   b5   |
;    \  /
;     b7
; preheader: 1000
; b2: 15
; b3: 7
; b4: 7
; Sink load to b2
; CHECK: t1
; CHECK: .b2:
; CHECK: load i32, i32* @g
; CHECK: .b3:
; CHECK-NOT:  load i32, i32* @g
define i32 @t1(i32, i32) #0 !prof !0 {
  %3 = icmp eq i32 %1, 0
  br i1 %3, label %.exit, label %.preheader

.preheader:
  %invariant = load i32, i32* @g
  br label %.b1

.b1:
  %iv = phi i32 [ %t7, %.b7 ], [ 0, %.preheader ]
  %c1 = icmp sgt i32 %iv, %0
  br i1 %c1, label %.b2, label %.b6, !prof !1

.b2:
  %c2 = icmp sgt i32 %iv, 1
  br i1 %c2, label %.b3, label %.b4

.b3:
  %t3 = sub nsw i32 %invariant, %iv
  br label %.b5

.b4:
  %t4 = add nsw i32 %invariant, %iv
  br label %.b5

.b5:
  %p5 = phi i32 [ %t3, %.b3 ], [ %t4, %.b4 ]
  %t5 = mul nsw i32 %p5, 5
  br label %.b7

.b6:
  %t6 = add nsw i32 %iv, 100
  br label %.b7

.b7:
  %p7 = phi i32 [ %t6, %.b6 ], [ %t5, %.b5 ]
  %t7 = add nuw nsw i32 %iv, 1
  %c7 = icmp eq i32 %t7, %p7
  br i1 %c7, label %.b1, label %.exit, !prof !3

.exit:
  ret i32 10
}

;     b1
;    /  \
;   b2  b6
;  /  \  |
; b3  b4 |
;  \  /  |
;   b5   |
;    \  /
;     b7
; preheader: 500
; b1: 16016
; b3: 8
; b6: 8
; Sink load to b3 and b6
; CHECK: t2
; CHECK: .preheader:
; CHECK-NOT: load i32, i32* @g
; CHECK: .b3:
; CHECK: load i32, i32* @g
; CHECK: .b4:
; CHECK: .b6:
; CHECK: load i32, i32* @g
; CHECK: .b7:
define i32 @t2(i32, i32) #0 !prof !0 {
  %3 = icmp eq i32 %1, 0
  br i1 %3, label %.exit, label %.preheader

.preheader:
  %invariant = load i32, i32* @g
  br label %.b1

.b1:
  %iv = phi i32 [ %t7, %.b7 ], [ 0, %.preheader ]
  %c1 = icmp sgt i32 %iv, %0
  br i1 %c1, label %.b2, label %.b6, !prof !2

.b2:
  %c2 = icmp sgt i32 %iv, 1
  br i1 %c2, label %.b3, label %.b4, !prof !1

.b3:
  %t3 = sub nsw i32 %invariant, %iv
  br label %.b5

.b4:
  %t4 = add nsw i32 5, %iv
  br label %.b5

.b5:
  %p5 = phi i32 [ %t3, %.b3 ], [ %t4, %.b4 ]
  %t5 = mul nsw i32 %p5, 5
  br label %.b7

.b6:
  %t6 = add nsw i32 %iv, %invariant
  br label %.b7

.b7:
  %p7 = phi i32 [ %t6, %.b6 ], [ %t5, %.b5 ]
  %t7 = add nuw nsw i32 %iv, 1
  %c7 = icmp eq i32 %t7, %p7
  br i1 %c7, label %.b1, label %.exit, !prof !3

.exit:
  ret i32 10
}

;     b1
;    /  \
;   b2  b6
;  /  \  |
; b3  b4 |
;  \  /  |
;   b5   |
;    \  /
;     b7
; preheader: 500
; b3: 8
; b5: 16008
; Do not sink load from preheader.
; CHECK: t3
; CHECK: .preheader:
; CHECK: load i32, i32* @g
; CHECK: .b1:
; CHECK-NOT: load i32, i32* @g
define i32 @t3(i32, i32) #0 !prof !0 {
  %3 = icmp eq i32 %1, 0
  br i1 %3, label %.exit, label %.preheader

.preheader:
  %invariant = load i32, i32* @g
  br label %.b1

.b1:
  %iv = phi i32 [ %t7, %.b7 ], [ 0, %.preheader ]
  %c1 = icmp sgt i32 %iv, %0
  br i1 %c1, label %.b2, label %.b6, !prof !2

.b2:
  %c2 = icmp sgt i32 %iv, 1
  br i1 %c2, label %.b3, label %.b4, !prof !1

.b3:
  %t3 = sub nsw i32 %invariant, %iv
  br label %.b5

.b4:
  %t4 = add nsw i32 5, %iv
  br label %.b5

.b5:
  %p5 = phi i32 [ %t3, %.b3 ], [ %t4, %.b4 ]
  %t5 = mul nsw i32 %p5, %invariant
  br label %.b7

.b6:
  %t6 = add nsw i32 %iv, 5
  br label %.b7

.b7:
  %p7 = phi i32 [ %t6, %.b6 ], [ %t5, %.b5 ]
  %t7 = add nuw nsw i32 %iv, 1
  %c7 = icmp eq i32 %t7, %p7
  br i1 %c7, label %.b1, label %.exit, !prof !3

.exit:
  ret i32 10
}

; For single-BB loop with <=1 avg trip count, sink load to b1
; CHECK: t4
; CHECK: .preheader:
; CHECK-not: load i32, i32* @g
; CHECK: .b1:
; CHECK: load i32, i32* @g
; CHECK: .exit:
define i32 @t4(i32, i32) #0 !prof !0 {
.preheader:
  %invariant = load i32, i32* @g
  br label %.b1

.b1:
  %iv = phi i32 [ %t1, %.b1 ], [ 0, %.preheader ]
  %t1 = add nsw i32 %invariant, %iv
  %c1 = icmp sgt i32 %iv, %0
  br i1 %c1, label %.b1, label %.exit, !prof !1

.exit:
  ret i32 10
}

;     b1
;    /  \
;   b2  b6
;  /  \  |
; b3  b4 |
;  \  /  |
;   b5   |
;    \  /
;     b7
; preheader: 1000
; b2: 15
; b3: 7
; b4: 7
; There is alias store in loop, do not sink load
; CHECK: t5
; CHECK: .preheader:
; CHECK: load i32, i32* @g
; CHECK: .b1:
; CHECK-NOT: load i32, i32* @g
define i32 @t5(i32, i32*) #0 !prof !0 {
  %3 = icmp eq i32 %0, 0
  br i1 %3, label %.exit, label %.preheader

.preheader:
  %invariant = load i32, i32* @g
  br label %.b1

.b1:
  %iv = phi i32 [ %t7, %.b7 ], [ 0, %.preheader ]
  %c1 = icmp sgt i32 %iv, %0
  br i1 %c1, label %.b2, label %.b6, !prof !1

.b2:
  %c2 = icmp sgt i32 %iv, 1
  br i1 %c2, label %.b3, label %.b4

.b3:
  %t3 = sub nsw i32 %invariant, %iv
  br label %.b5

.b4:
  %t4 = add nsw i32 %invariant, %iv
  br label %.b5

.b5:
  %p5 = phi i32 [ %t3, %.b3 ], [ %t4, %.b4 ]
  %t5 = mul nsw i32 %p5, 5
  br label %.b7

.b6:
  %t6 = call i32 @foo()
  br label %.b7

.b7:
  %p7 = phi i32 [ %t6, %.b6 ], [ %t5, %.b5 ]
  %t7 = add nuw nsw i32 %iv, 1
  %c7 = icmp eq i32 %t7, %p7
  br i1 %c7, label %.b1, label %.exit, !prof !3

.exit:
  ret i32 10
}

declare i32 @foo()

!0 = !{!"function_entry_count", i64 1}
!1 = !{!"branch_weights", i32 1, i32 2000}
!2 = !{!"branch_weights", i32 2000, i32 1}
!3 = !{!"branch_weights", i32 100, i32 1}
