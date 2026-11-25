; RUN: opt -passes='cgscc(inline)' -pass-remarks=inline -pass-remarks-missed=inline -S %s -o /dev/null 2>&1 | FileCheck %s
; RUN: opt -passes='cgscc(inline)' -pass-remarks=inline -inline-caller-loop-nest-penalty-scale=0 -S %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=NOPENALTY
; CHECK: 'large_loop_body' not inlined into 'caller_nested_loop' because too costly to inline
; CHECK: 'small_loop_body' inlined into 'caller_nested_loop'
;
; NOPENALTY: 'large_loop_body' inlined into 'caller_nested_loop'

define void @caller_nested_loop(ptr noalias %A, ptr noalias %B, double %scalar, i32 %N, i32 %M) {
entry:
  %cmp_outer = icmp sgt i32 %N, 0
  br i1 %cmp_outer, label %outer.header, label %exit

outer.header:
  %i = phi i32 [ 0, %entry ], [ %i.next, %outer.latch ]
  %cmp_inner = icmp sgt i32 %M, 0
  br i1 %cmp_inner, label %inner.header, label %outer.latch

inner.header:
  %j = phi i32 [ 0, %outer.header ], [ %j.next, %inner.latch ]
  ; Compute column pointer offsets
  %j_ext = sext i32 %j to i64
  %i_ext = sext i32 %i to i64
  %off_a = mul nsw i64 %j_ext, 200
  %off_b = mul nsw i64 %i_ext, 200
  %pa = getelementptr inbounds double, ptr %A, i64 %off_a
  %pb = getelementptr inbounds double, ptr %B, i64 %off_b
  %len = sub nsw i32 %N, %i
  call void @large_loop_body(ptr %pa, ptr %pb, double %scalar, i32 %len)
  call void @small_loop_body(ptr %pa, ptr %pb, double %scalar, i32 %len)
  br label %inner.latch

inner.latch:
  %j.next = add nuw nsw i32 %j, 1
  %cmp_j = icmp slt i32 %j.next, %M
  br i1 %cmp_j, label %inner.header, label %outer.latch, !prof !0

outer.latch:
  %i.next = add nuw nsw i32 %i, 1
  %cmp_i = icmp slt i32 %i.next, %N
  br i1 %cmp_i, label %outer.header, label %exit, !prof !1

exit:
  ret void
}

define void @large_loop_body(ptr %dst, ptr %src, double %da, i32 %n) {
entry:
  %cmp = icmp sgt i32 %n, 3
  br i1 %cmp, label %loop.preheader, label %exit

loop.preheader:
  %n.ext = sext i32 %n to i64
  br label %loop

loop:
  %iv = phi i64 [ 0, %loop.preheader ], [ %iv.next, %loop ]

  %src.p0 = getelementptr inbounds double, ptr %src, i64 %iv
  %dst.p0 = getelementptr inbounds double, ptr %dst, i64 %iv
  %s0 = load double, ptr %src.p0, align 8
  %d0 = load double, ptr %dst.p0, align 8
  %m0 = fmul fast double %s0, %da
  %a0 = fadd fast double %m0, %d0
  store double %a0, ptr %dst.p0, align 8

  %iv1 = add nuw nsw i64 %iv, 1
  %src.p1 = getelementptr inbounds double, ptr %src, i64 %iv1
  %dst.p1 = getelementptr inbounds double, ptr %dst, i64 %iv1
  %s1 = load double, ptr %src.p1, align 8
  %d1 = load double, ptr %dst.p1, align 8
  %m1 = fmul fast double %s1, %da
  %a1 = fadd fast double %m1, %d1
  store double %a1, ptr %dst.p1, align 8

  %iv2 = add nuw nsw i64 %iv, 2
  %src.p2 = getelementptr inbounds double, ptr %src, i64 %iv2
  %dst.p2 = getelementptr inbounds double, ptr %dst, i64 %iv2
  %s2 = load double, ptr %src.p2, align 8
  %d2 = load double, ptr %dst.p2, align 8
  %m2 = fmul fast double %s2, %da
  %a2 = fadd fast double %m2, %d2
  store double %a2, ptr %dst.p2, align 8

  %iv3 = add nuw nsw i64 %iv, 3
  %src.p3 = getelementptr inbounds double, ptr %src, i64 %iv3
  %dst.p3 = getelementptr inbounds double, ptr %dst, i64 %iv3
  %s3 = load double, ptr %src.p3, align 8
  %d3 = load double, ptr %dst.p3, align 8
  %m3 = fmul fast double %s3, %da
  %a3 = fadd fast double %m3, %d3
  store double %a3, ptr %dst.p3, align 8

  %iv.next = add nuw nsw i64 %iv, 4
  %cmp.loop = icmp slt i64 %iv.next, %n.ext
  br i1 %cmp.loop, label %loop, label %exit

exit:
  ret void
}


define void @small_loop_body(ptr %dst, ptr %src, double %da, i32 %n) {
entry:
  %cmp = icmp sgt i32 %n, 0
  br i1 %cmp, label %loop.preheader, label %exit

loop.preheader:
  %n.ext = sext i32 %n to i64
  br label %loop

loop:
  %iv = phi i64 [ 0, %loop.preheader ], [ %iv.next, %loop ]
  %src.ptr = getelementptr inbounds double, ptr %src, i64 %iv
  %dst.ptr = getelementptr inbounds double, ptr %dst, i64 %iv
  %s = load double, ptr %src.ptr, align 8
  %d = load double, ptr %dst.ptr, align 8
  %m = fmul fast double %s, %da
  %a = fadd fast double %m, %d
  store double %a, ptr %dst.ptr, align 8
  %iv.next = add nuw nsw i64 %iv, 1
  %cmp.loop = icmp slt i64 %iv.next, %n.ext
  br i1 %cmp.loop, label %loop, label %exit

exit:
  ret void
}

!0 = !{!"branch_weights", i32 199, i32 1}
!1 = !{!"branch_weights", i32 199, i32 1}
