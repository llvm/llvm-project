; REQUIRES: asserts

; RUN: opt -passes=loop-vectorize -pass-remarks=loop-vectorize -enable-vplan-native-path -debug -disable-output %s 2>&1 | FileCheck %s

@arr2 = external global [8 x i64], align 16
@arr = external global [8 x [8 x i64]], align 16

define void @foo(i64 %n) {
; CHECK:      VPlan 'Plain CFG
; CHECK-NEXT: {
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<entry>:
; CHECK-NEXT: Successor(s): outer.header
; CHECK-EMPTY:
; CHECK-NEXT: outer.header:
; CHECK-NEXT:   EMIT-SCALAR ir<%outer.iv> = phi [ ir<%outer.iv.next>, outer.latch ], [ ir<0>, ir-bb<entry> ]
; CHECK-NEXT:   EMIT ir<%gep.1> = getelementptr ir<@arr2>, ir<0>, ir<%outer.iv>
; CHECK-NEXT:   EMIT store ir<%outer.iv>, ir<%gep.1>
; CHECK-NEXT:   EMIT ir<%add> = add ir<%outer.iv>, ir<%n>
; CHECK-NEXT: Successor(s): inner
; CHECK-EMPTY:
; CHECK-NEXT: inner:
; CHECK-NEXT:   EMIT-SCALAR ir<%inner.iv> = phi [ ir<%inner.iv.next>, inner ], [ ir<0>, outer.header ]
; CHECK-NEXT:   EMIT ir<%gep.2> = getelementptr ir<@arr>, ir<0>, ir<%inner.iv>, ir<%outer.iv>
; CHECK-NEXT:   EMIT store ir<%add>, ir<%gep.2>
; CHECK-NEXT:   EMIT ir<%inner.iv.next> = add ir<%inner.iv>, ir<1>
; CHECK-NEXT:   EMIT ir<%inner.ec> = icmp ir<%inner.iv.next>, ir<8>
; CHECK-NEXT:   EMIT branch-on-cond ir<%inner.ec>
; CHECK-NEXT: Successor(s): outer.latch, inner
; CHECK-EMPTY:
; CHECK-NEXT: outer.latch:
; CHECK-NEXT:   EMIT ir<%outer.iv.next> = add ir<%outer.iv>, ir<1>
; CHECK-NEXT:   EMIT ir<%outer.ec> = icmp ir<%outer.iv.next>, ir<8>
; CHECK-NEXT:   EMIT branch-on-cond ir<%outer.ec>
; CHECK-NEXT: Successor(s): ir-bb<exit>, outer.header
; CHECK-EMPTY:
; CHECK-NEXT: ir-bb<exit>:
; CHECK-NEXT: No successors
; CHECK-NEXT: }
entry:
  br label %outer.header

outer.header:
  %outer.iv = phi i64 [ 0, %entry ], [ %outer.iv.next, %outer.latch ]
  %gep.1 = getelementptr inbounds [8 x i64], ptr @arr2, i64 0, i64 %outer.iv
  store i64 %outer.iv, ptr %gep.1, align 4
  %add = add nsw i64 %outer.iv, %n
  br label %inner

inner:
  %inner.iv = phi i64 [ 0, %outer.header ], [ %inner.iv.next, %inner ]
  %gep.2 = getelementptr inbounds [8 x [8 x i64]], ptr @arr, i64 0, i64 %inner.iv, i64 %outer.iv
  store i64 %add, ptr %gep.2, align 4
  %inner.iv.next = add nuw nsw i64 %inner.iv, 1
  %inner.ec = icmp eq i64 %inner.iv.next, 8
  br i1 %inner.ec, label %outer.latch, label %inner

outer.latch:
  %outer.iv.next = add nuw nsw i64 %outer.iv, 1
  %outer.ec = icmp eq i64 %outer.iv.next, 8
  br i1 %outer.ec, label %exit, label %outer.header, !llvm.loop !1

exit:
  ret void
}

; CHECK: remark: <unknown>:0:0: vectorized outer loop (vectorization width: 4, interleaved count: 1)

!1 = distinct !{!1, !2, !3}
!2 = !{!"llvm.loop.vectorize.width", i32 4}
!3 = !{!"llvm.loop.vectorize.enable", i1 true}
