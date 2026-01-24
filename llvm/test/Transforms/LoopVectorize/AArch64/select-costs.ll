; REQUIRES: asserts
; RUN: opt < %s -passes=loop-vectorize -debug-only=loop-vectorize -disable-output -S 2>&1 | FileCheck %s

target datalayout = "e-m:o-i64:64-i128:128-n32:64-S128"
target triple = "arm64-apple-ios5.0.0"

define void @selects_1(ptr nocapture %dst, i32 %A, i32 %B, i32 %C, i32 %N) {
; CHECK: LV: Checking a loop in 'selects_1'

; CHECK: Cost of 1 for VF 2: WIDEN ir<%cond> = select ir<%cmp1>, ir<10>, ir<%and>
; CHECK: Cost of 1 for VF 2: WIDEN ir<%cond6> = select ir<%cmp2>, ir<30>, ir<%and>
; CHECK: Cost of 1 for VF 2: WIDEN ir<%cond11> = select ir<%cmp7>, ir<%cond>, ir<%cond6>

; CHECK: Cost of 1 for VF 4: WIDEN ir<%cond> = select ir<%cmp1>, ir<10>, ir<%and>
; CHECK: Cost of 1 for VF 4: WIDEN ir<%cond6> = select ir<%cmp2>, ir<30>, ir<%and>
; CHECK: Cost of 1 for VF 4: WIDEN ir<%cond11> = select ir<%cmp7>, ir<%cond>, ir<%cond6>

; CHECK: LV: Selecting VF: 4

entry:
  %n = zext i32 %N to i64
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %arrayidx = getelementptr inbounds i32, ptr %dst, i64 %iv
  %0 = load i32, ptr %arrayidx, align 4
  %and = and i32 %0, 2047
  %cmp1 = icmp eq i32 %and, %A
  %cond = select i1 %cmp1, i32 10, i32 %and
  %cmp2 = icmp eq i32 %and, %B
  %cond6 = select i1 %cmp2, i32 30, i32 %and
  %cmp7 = icmp ugt i32 %cond, %C
  %cond11 = select i1 %cmp7, i32 %cond, i32 %cond6
  store i32 %cond11, ptr %arrayidx, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %n
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

define i32 @multi_user_cmp(ptr readonly %a, i64 noundef %n) {
; CHECK: LV: Checking a loop in 'multi_user_cmp'
; CHECK: Cost of 4 for VF 16: WIDEN ir<%cmp1> = fcmp olt ir<%load1>, ir<0.000000e+00>
; CHECK: LV: Selecting VF: 16.
entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %all.off.next = phi i1 [ true, %entry ], [ %all.off, %loop ]
  %any.0.off09 = phi i1 [ false, %entry ], [ %.any.0.off0, %loop ]
  %arrayidx = getelementptr inbounds float, ptr %a, i64 %iv
  %load1 = load float, ptr %arrayidx, align 4
  %cmp1 = fcmp olt float %load1, 0.000000e+00
  %.any.0.off0 = select i1 %cmp1, i1 true, i1 %any.0.off09
  %all.off = select i1 %cmp1, i1 %all.off.next, i1 false
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %n
  br i1 %ec, label %exit, label %loop

exit:
  %0 = select i1 %.any.0.off0, i32 2, i32 3
  %1 = select i1 %all.off, i32 1, i32 %0
  ret i32 %1
}

define i32 @select_vpinst_for_tail_folding(i8 %n) {
; CHECK: LV: Checking a loop in 'select_vpinst_for_tail_folding'
; CHECK: Cost of 1 for VF 2: EMIT vp<{{.+}}> = select vp<{{.+}}>, ir<%red.next>, ir<%red>
; CHECK: Cost of 1 for VF 4: EMIT vp<{{.+}}> = select vp<{{.+}}>, ir<%red.next>, ir<%red>
; CHECK: LV: Selecting VF: 4

entry:
  %c = icmp ne i8 %n, 0
  %ext = zext i1 %c to i32
  br label %loop

loop:
  %iv = phi i32 [ %ext, %entry ], [ %iv.next, %loop ]
  %red = phi i32 [ 0, %entry ], [ %red.next, %loop ]
  %iv.next = add i32 %iv, 1
  %red.next = mul i32 %red, %iv
  %ec = icmp eq i32 %iv, 12
  br i1 %ec, label %exit, label %loop

exit:
  ret i32 %red.next
}

define i32 @select_xor_cond(ptr %src, i1 %c.0) {
; CHECK: LV: Checking a loop in 'select_xor_cond'
; CHECK: Cost of 1 for VF 2: WIDEN ir<%sel> = select ir<%c>, ir<false>, ir<%c.0>
; CHECK: Cost of 1 for VF 4: WIDEN ir<%sel> = select ir<%c>, ir<false>, ir<%c.0>
; CHECK: Cost of 1 for VF 8: WIDEN ir<%sel> = select ir<%c>, ir<false>, ir<%c.0>
; CHECK: Cost of 1 for VF 16: WIDEN ir<%sel> = select ir<%c>, ir<false>, ir<%c.0>
; CHECK: LV: Selecting VF: 4.

entry:
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %p = phi i32 [ 0, %entry ], [ %ext.sel, %loop ]
  %gep.src = getelementptr i8, ptr %src, i64 %iv
  %0 = load i8, ptr %gep.src, align 1
  %c = icmp eq i8 %0, 0
  %not.c = xor i1 %c, true
  %sel = select i1 %not.c, i1 %c.0, i1 false
  %ext.sel = zext i1 %sel to i32
  %iv.next = add i64 %iv, 1
  %ec = icmp eq i64 %iv, 19
  br i1 %ec, label %exit, label %loop

exit:
  %ext.c = zext i1 %c to i32
  %res = add i32 %ext.c, %p
  ret i32 %res
}

define void @select_invariant_cmp_cond(ptr %dst, ptr %src, i32 %a, i32 %b, i64 %n) "target-cpu"="neoverse-v2" {
; CHECK: LV: Checking a loop in 'select_invariant_cmp_cond'
; CHECK: Cost of 1 for VF 2: WIDEN ir<%sel> = select ir<%cmp>, ir<%trunc>, ir<0>
; CHECK: Cost of 1 for VF 4: WIDEN ir<%sel> = select ir<%cmp>, ir<%trunc>, ir<0>
; CHECK: LV: Selecting VF: 4.
entry:
  %cmp = icmp ugt i32 %a, %b
  br label %loop

loop:
  %iv = phi i64 [ 0, %entry ], [ %iv.next, %loop ]
  %gep.src = getelementptr inbounds i32, ptr %src, i64 %iv
  %load = load i32, ptr %gep.src, align 4
  %conv = sext i32 %load to i64
  %or = or i64 %conv, 1
  %trunc = trunc i64 %or to i32
  %sel = select i1 %cmp, i32 %trunc, i32 0
  %gep.dst = getelementptr inbounds i32, ptr %dst, i64 %iv
  store i32 %sel, ptr %gep.dst, align 4
  %iv.next = add nuw nsw i64 %iv, 1
  %ec = icmp eq i64 %iv.next, %n
  br i1 %ec, label %exit, label %loop

exit:
  ret void
}

declare void @use(i64, i32)

define void @logical_and_select_inverted(i1 %cmp, ptr %start, ptr %end) {
; CHECK: LV: Checking a loop in 'logical_and_select_inverted'
; CHECK: Cost of 1 for VF 2: WIDEN ir<%narrow> = select ir<%trunc>, ir<false>, ir<%cmp>
; CHECK: Cost of 1 for VF 4: WIDEN ir<%narrow> = select ir<%trunc>, ir<false>, ir<%cmp>
; CHECK: Cost of 1 for VF 8: WIDEN ir<%narrow> = select ir<%trunc>, ir<false>, ir<%cmp>
; CHECK: Cost of 1 for VF 16: WIDEN ir<%narrow> = select ir<%trunc>, ir<false>, ir<%cmp>
; CHECK: LV: Selecting VF: 16.

entry:
  br label %loop

loop:
  %for = phi i64 [ %zext, %loop ], [ 0, %entry ]
  %p = phi i32 [ %spec.select, %loop ], [ 0, %entry ]
  %ptr = phi ptr [ %ptr.next, %loop ], [ %start, %entry ]
  %ld = load i8, ptr %ptr, align 8
  %trunc = trunc i8 %ld to i1
  %not = xor i1 %trunc, true
  %zext = zext i1 %not to i64
  %or.1 = or i64 %for, 1
  %narrow = select i1 %not, i1 %cmp, i1 false
  %spec.select = zext i1 %narrow to i32
  %or.2 = or i32 %p, 1
  %ptr.next = getelementptr i8, ptr %ptr, i64 40
  %ec = icmp eq ptr %ptr.next, %end
  br i1 %ec, label %exit, label %loop

exit:
  call void @use(i64 %or.1, i32 %or.2)
  ret void
}

define void @logical_or_select_inverted(i1 %cmp, ptr %start, ptr %end) {
; CHECK: LV: Checking a loop in 'logical_or_select_inverted'
; CHECK: Cost of 1 for VF 2: WIDEN ir<%narrow> = select ir<%trunc>, ir<%cmp>, ir<true>
; CHECK: Cost of 1 for VF 4: WIDEN ir<%narrow> = select ir<%trunc>, ir<%cmp>, ir<true>
; CHECK: Cost of 1 for VF 8: WIDEN ir<%narrow> = select ir<%trunc>, ir<%cmp>, ir<true>
; CHECK: Cost of 1 for VF 16: WIDEN ir<%narrow> = select ir<%trunc>, ir<%cmp>, ir<true>
; CHECK: LV: Selecting VF: 16.

entry:
  br label %loop

loop:
  %for = phi i64 [ %zext, %loop ], [ 0, %entry ]
  %p = phi i32 [ %spec.select, %loop ], [ 0, %entry ]
  %ptr = phi ptr [ %ptr.next, %loop ], [ %start, %entry ]
  %ld = load i8, ptr %ptr, align 8
  %trunc = trunc i8 %ld to i1
  %not = xor i1 %trunc, true
  %zext = zext i1 %not to i64
  %or.1 = or i64 %for, 1
  %narrow = select i1 %not, i1 true, i1 %cmp
  %spec.select = zext i1 %narrow to i32
  %or.2 = or i32 %p, 1
  %ptr.next = getelementptr i8, ptr %ptr, i64 40
  %ec = icmp eq ptr %ptr.next, %end
  br i1 %ec, label %exit, label %loop

exit:
  call void @use(i64 %or.1, i32 %or.2)
  ret void
}
