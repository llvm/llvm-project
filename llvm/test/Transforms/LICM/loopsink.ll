; RUN: opt -S -verify-memoryssa -passes=loop-sink < %s | FileCheck %s
; RUN: opt -S -verify-memoryssa -aa-pipeline=basic-aa -passes=loop-sink < %s | FileCheck %s

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
; CHECK: load i32, ptr @g
; CHECK: .b3:
; CHECK-NOT:  load i32, ptr @g
define i32 @t1(i32, i32) #0 !prof !0 {
  %3 = icmp eq i32 %1, 0
  br i1 %3, label %.exit, label %.preheader

.preheader:
  %invariant = load i32, ptr @g
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
; CHECK-NOT: load i32, ptr @g
; CHECK: .b3:
; CHECK: load i32, ptr @g
; CHECK: .b4:
; CHECK: .b6:
; CHECK: load i32, ptr @g
; CHECK: .b7:
define i32 @t2(i32, i32) #0 !prof !0 {
  %3 = icmp eq i32 %1, 0
  br i1 %3, label %.exit, label %.preheader

.preheader:
  %invariant = load i32, ptr @g
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
; CHECK: load i32, ptr @g
; CHECK: .b1:
; CHECK-NOT: load i32, ptr @g
define i32 @t3(i32, i32) #0 !prof !0 {
  %3 = icmp eq i32 %1, 0
  br i1 %3, label %.exit, label %.preheader

.preheader:
  %invariant = load i32, ptr @g
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

; For single-BB loop with <=1 avg trip count, sink load to body
; CHECK: t4
; CHECK: .header:
; CHECK-NOT: load i32, ptr @g
; CHECK: .body:
; CHECK: load i32, ptr @g
; CHECK: .exit:
define i32 @t4(i32, i32) #0 !prof !0 {
.entry:
  %invariant = load i32, ptr @g
  br label %.header

.header:
  %iv = phi i32 [ %t1, %.body ], [ 0, %.entry ]
  %c0 = icmp sgt i32 %iv, %0
  br i1 %c0, label %.body, label %.exit, !prof !1

.body:
  %t1 = add nsw i32 %invariant, %iv
  %c1 = icmp sgt i32 %iv, %0
  br label %.header

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
; CHECK: load i32, ptr @g
; CHECK: .b1:
; CHECK-NOT: load i32, ptr @g
define i32 @t5(i32, ptr) #0 !prof !0 {
  %3 = icmp eq i32 %0, 0
  br i1 %3, label %.exit, label %.preheader

.preheader:
  %invariant = load i32, ptr @g
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
; Regardless of aliasing store in loop this load from constant memory can be sunk.
; CHECK: t5_const_memory
; CHECK: .preheader:
; CHECK-NOT: load i32, ptr @g_const
; CHECK: .b2:
; CHECK: load i32, ptr @g_const
; CHECK: br i1 %c2, label %.b3, label %.b4
define i32 @t5_const_memory(i32, ptr) #0 !prof !0 {
  %3 = icmp eq i32 %0, 0
  br i1 %3, label %.exit, label %.preheader

.preheader:
  %invariant = load i32, ptr @g_const
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

;     b1
;    /  \
;   b2  b3
;    \  /
;     b4
; preheader: 1000
; b2: 15
; b3: 7
; Do not sink unordered atomic load to b2
; CHECK: t6
; CHECK: .preheader:
; CHECK:  load atomic i32, ptr @g unordered, align 4
; CHECK: .b2:
; CHECK-NOT: load atomic i32, ptr @g unordered, align 4
define i32 @t6(i32, i32) #0 !prof !0 {
  %3 = icmp eq i32 %1, 0
  br i1 %3, label %.exit, label %.preheader

.preheader:
  %invariant = load atomic i32, ptr @g unordered, align 4
  br label %.b1

.b1:
  %iv = phi i32 [ %t3, %.b4 ], [ 0, %.preheader ]
  %c1 = icmp sgt i32 %iv, %0
  br i1 %c1, label %.b2, label %.b3, !prof !1

.b2:
  %t1 = add nsw i32 %invariant, %iv
  br label %.b4

.b3:
  %t2 = add nsw i32 %iv, 100
  br label %.b4

.b4:
  %p1 = phi i32 [ %t2, %.b3 ], [ %t1, %.b2 ]
  %t3 = add nuw nsw i32 %iv, 1
  %c2 = icmp eq i32 %t3, %p1
  br i1 %c2, label %.b1, label %.exit, !prof !3

.exit:
  ret i32 10
}

@g_const = constant i32 0, align 4

;     b1
;    /  \
;   b2  b3
;    \  /
;     b4
; preheader: 1000
; b2: 0.5
; b3: 999.5
; Sink unordered atomic load to b2. It is allowed to sink into loop unordered
; load from constant.
; CHECK: t7
; CHECK: .preheader:
; CHECK-NOT:  load atomic i32, ptr @g_const unordered, align 4
; CHECK: .b2:
; CHECK: load atomic i32, ptr @g_const unordered, align 4
define i32 @t7(i32, i32) #0 !prof !0 {
  %3 = icmp eq i32 %1, 0
  br i1 %3, label %.exit, label %.preheader

.preheader:
  %invariant = load atomic i32, ptr @g_const unordered, align 4
  br label %.b1

.b1:
  %iv = phi i32 [ %t3, %.b4 ], [ 0, %.preheader ]
  %c1 = icmp sgt i32 %iv, %0
  br i1 %c1, label %.b2, label %.b3, !prof !1

.b2:
  %t1 = add nsw i32 %invariant, %iv
  br label %.b4

.b3:
  %t2 = add nsw i32 %iv, 100
  br label %.b4

.b4:
  %p1 = phi i32 [ %t2, %.b3 ], [ %t1, %.b2 ]
  %t3 = add nuw nsw i32 %iv, 1
  %c2 = icmp eq i32 %t3, %p1
  br i1 %c2, label %.b1, label %.exit, !prof !3

.exit:
  ret i32 10
}

; CHECK-LABEL: @sink_earliest_capture(
; CHECK: entry:
; CHECK: %r0 = call i32 @w(ptr %a)
; CHECK: cold:
; CHECK-NOT: %r0 = call i32 @w(ptr %a)
; CHECK: %x = call ptr @opaque(ptr %a)
define i32 @sink_earliest_capture(ptr noalias %p, i32 %n) !prof !0 {
entry:
  %a = alloca i32, align 4
  store i32 0, ptr %a, align 4
  %r0 = call i32 @w(ptr %a)
  %x = call ptr @opaque(ptr %a)
  store ptr %a, ptr %p, align 8
  %q = load ptr, ptr %p, align 8
  %r2 = call i32 @w(ptr %a)
  br label %loop

loop:
  %iv = phi i32 [ 0, %entry ], [ %iv.next, %latch ]
  store i32 1, ptr %q, align 4
  %c1 = icmp eq i32 %iv, 7
  br i1 %c1, label %cold, label %latch, !prof !1

cold:
  call void @use(i32 %r0)
  call void @use.p(ptr %x)
  call void @use(i32 %r2)
  br label %latch

latch:
  %iv.next = add nuw nsw i32 %iv, 1
  %c = icmp ne i32 %iv.next, %n
  br i1 %c, label %loop, label %exit, !prof !3

exit:
  %v = load i32, ptr %a, align 4
  ret i32 %v
}

declare i32 @foo()
declare ptr @opaque(ptr) memory(none) nounwind willreturn
declare i32 @w(ptr captures(none)) memory(argmem: write) nounwind willreturn
declare void @use(i32) memory(inaccessiblemem: write) nounwind willreturn
declare void @use.p(ptr) memory(inaccessiblemem: write) nounwind willreturn

!0 = !{!"function_entry_count", i64 1}
!1 = !{!"branch_weights", i32 1, i32 2000}
!2 = !{!"branch_weights", i32 2000, i32 1}
!3 = !{!"branch_weights", i32 100, i32 1}
