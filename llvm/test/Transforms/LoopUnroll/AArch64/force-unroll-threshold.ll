; RUN: opt -passes=loop-unroll -S -unroll-runtime %s | FileCheck %s --check-prefix=NOFORCE
; RUN: opt -passes=loop-unroll -S -unroll-runtime -aarch64-force-unroll-threshold=500 %s | FileCheck %s --check-prefix=FORCE

; The loop has a small runtime upper bound (at most four iterations) but a
; relatively expensive body. With runtime unrolling enabled, the cost model
; still leaves the loop rolled. Raising the AArch64 force threshold overrides
; that decision and unrolls.

target triple = "aarch64-unknown-linux-gnu"

define void @force_small_loop(ptr nocapture %a, ptr nocapture %b, i32 %n) {
entry:
  br label %loop

; NOFORCE-LABEL: @force_small_loop(
; NOFORCE:       loop:
; NOFORCE:         br i1 %cond, label %body, label %exit
; NOFORCE:       body:
; NOFORCE:         store i32 %mix15, ptr %ptrb, align 4
; NOFORCE:       latch:
; NOFORCE:         br i1 %cmp2, label %loop, label %exit
; NOFORCE:       ret void
; NOFORCE-NOT:   loop.1:
;
; FORCE-LABEL: @force_small_loop(
; FORCE:       loop:
; FORCE:         br i1 %cond, label %body, label %exit
; FORCE:       loop.1:
; FORCE:         br i1 true, label %body.1, label %exit
; FORCE:       body.1:
; FORCE:         store i32 %mix15.1, ptr %ptrb.1, align 4
; FORCE:       latch.1:
; FORCE:         br i1 %cmp2.1, label %loop, label %exit
; FORCE:       ret void

loop:
  %i = phi i32 [ 0, %entry ], [ %inc, %latch ]
  %ptra = getelementptr inbounds i32, ptr %a, i32 %i
  %pa = load i32, ptr %ptra, align 4
  %tmp0 = mul nsw i32 %pa, %pa
  %tmp1 = add nsw i32 %tmp0, %pa
  %tmp2 = shl i32 %tmp1, 1
  %tmp3 = ashr i32 %tmp2, 1
  %tmp4 = xor i32 %tmp3, %pa
  %tmp5 = add nsw i32 %tmp4, 7
  %tmp6 = mul nsw i32 %tmp5, 5
  %tmp7 = add nsw i32 %tmp6, %tmp4
  %tmp8 = mul nsw i32 %tmp7, %tmp3
  %tmp9 = add nsw i32 %tmp8, %tmp7
  %tmp10 = xor i32 %tmp9, %tmp6
  %tmp11 = add nsw i32 %tmp10, %tmp8
  %tmp12 = mul nsw i32 %tmp11, 9
  %tmp13 = add nsw i32 %tmp12, %tmp10
  %tmp14 = xor i32 %tmp13, %tmp11
  %cond = icmp ult i32 %i, %n
  br i1 %cond, label %body, label %exit

body:
  %ptrb = getelementptr inbounds i32, ptr %b, i32 %i
  %pb = load i32, ptr %ptrb, align 4
  %sum = add nsw i32 %pb, %tmp14
  %diff = sub nsw i32 %sum, %pa
  %mix1 = mul nsw i32 %diff, 3
  %mix2 = add nsw i32 %mix1, %tmp3
  %mix3 = xor i32 %mix2, %diff
  %mix4 = add nsw i32 %mix3, %tmp0
  %mix5 = mul nsw i32 %mix4, 11
  %mix6 = add nsw i32 %mix5, %mix2
  %mix7 = xor i32 %mix6, %mix5
  %mix8 = add nsw i32 %mix7, %mix3
  %mix9 = mul nsw i32 %mix8, 13
  %mix10 = add nsw i32 %mix9, %mix8
  %mix11 = xor i32 %mix10, %mix7
  %mix12 = add nsw i32 %mix11, %mix6
  %mix13 = mul nsw i32 %mix12, 17
  %mix14 = add nsw i32 %mix13, %mix9
  %mix15 = xor i32 %mix14, %mix10
  store i32 %mix15, ptr %ptrb, align 4
  br label %latch

latch:
  %inc = add nuw nsw i32 %i, 1
  %cmp.limit = icmp ult i32 %n, 4
  %upper = select i1 %cmp.limit, i32 %n, i32 4
  %cmp2 = icmp ult i32 %inc, %upper
  br i1 %cmp2, label %loop, label %exit

exit:
  ret void
}
