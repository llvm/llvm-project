; RUN: llc < %s -mtriple=x86_64-unknown-linux-gnu -mcpu=core2 -pre-RA-sched=source -enable-misched -verify-machineinstrs | FileCheck %s
;
; Verify that misched resource/latency balancy heuristics are sane.

define void @unrolled_mmult1(i32* %tmp55, i32* %tmp56, i32* %pre, i32* %pre94,
  i32* %pre95, i32* %pre96, i32* %pre97, i32* %pre98, i32* %pre99,
 i32* %pre100, i32* %pre101, i32* %pre102, i32* %pre103, i32* %pre104)
  nounwind uwtable ssp {
entry:
  br label %for.body

; imull folded loads should be in order, addl may be reordered to reduce total
; latency. Also check that we have no spilling.
;
; Since mmult1 IR is already in good order, this effectively ensure
; the scheduler maintains source order.
;
; CHECK-LABEL: %for.body
; CHECK-NOT: %rsp
; CHECK: imull 4
; CHECK-NOT: {{imull|rsp}}
; CHECK: addl
; CHECK: imull 8
; CHECK-NOT: {{imull|rsp}}
; CHECK: imull 12
; CHECK-NOT: {{imull|rsp}}
; CHECK: addl
; CHECK: addl
; CHECK: imull 16
; CHECK-NOT: {{imull|rsp}}
; CHECK: imull 20
; CHECK-NOT: {{imull|rsp}}
; CHECK: addl
; CHECK: imull 24
; CHECK-NOT: {{imull|rsp}}
; CHECK: addl
; CHECK: addl
; CHECK: imull 28
; CHECK-NOT: {{imull|rsp}}
; CHECK: imull 32
; CHECK-NOT: {{imull|rsp}}
; CHECK: addl
; CHECK: imull 36
; CHECK-NOT: {{imull|rsp}}
; CHECK: addl
; CHECK: addl
; CHECK-NOT: {{imull|rsp}}
; CHECK-LABEL: %end
for.body:
  %indvars.iv42.i = phi i64 [ %indvars.iv.next43.i, %for.body ], [ 0, %entry ]
  %tmp57 = load i32, i32* %tmp56, align 4
  %arrayidx12.us.i61 = getelementptr inbounds i32, i32* %pre, i64 %indvars.iv42.i
  %tmp58 = load i32, i32* %arrayidx12.us.i61, align 4
  %mul.us.i = mul nsw i32 %tmp58, %tmp57
  %arrayidx8.us.i.1 = getelementptr inbounds i32, i32* %tmp56, i64 1
  %tmp59 = load i32, i32* %arrayidx8.us.i.1, align 4
  %arrayidx12.us.i61.1 = getelementptr inbounds i32, i32* %pre94, i64 %indvars.iv42.i
  %tmp60 = load i32, i32* %arrayidx12.us.i61.1, align 4
  %mul.us.i.1 = mul nsw i32 %tmp60, %tmp59
  %add.us.i.1 = add nsw i32 %mul.us.i.1, %mul.us.i
  %arrayidx8.us.i.2 = getelementptr inbounds i32, i32* %tmp56, i64 2
  %tmp61 = load i32, i32* %arrayidx8.us.i.2, align 4
  %arrayidx12.us.i61.2 = getelementptr inbounds i32, i32* %pre95, i64 %indvars.iv42.i
  %tmp62 = load i32, i32* %arrayidx12.us.i61.2, align 4
  %mul.us.i.2 = mul nsw i32 %tmp62, %tmp61
  %add.us.i.2 = add nsw i32 %mul.us.i.2, %add.us.i.1
  %arrayidx8.us.i.3 = getelementptr inbounds i32, i32* %tmp56, i64 3
  %tmp63 = load i32, i32* %arrayidx8.us.i.3, align 4
  %arrayidx12.us.i61.3 = getelementptr inbounds i32, i32* %pre96, i64 %indvars.iv42.i
  %tmp64 = load i32, i32* %arrayidx12.us.i61.3, align 4
  %mul.us.i.3 = mul nsw i32 %tmp64, %tmp63
  %add.us.i.3 = add nsw i32 %mul.us.i.3, %add.us.i.2
  %arrayidx8.us.i.4 = getelementptr inbounds i32, i32* %tmp56, i64 4
  %tmp65 = load i32, i32* %arrayidx8.us.i.4, align 4
  %arrayidx12.us.i61.4 = getelementptr inbounds i32, i32* %pre97, i64 %indvars.iv42.i
  %tmp66 = load i32, i32* %arrayidx12.us.i61.4, align 4
  %mul.us.i.4 = mul nsw i32 %tmp66, %tmp65
  %add.us.i.4 = add nsw i32 %mul.us.i.4, %add.us.i.3
  %arrayidx8.us.i.5 = getelementptr inbounds i32, i32* %tmp56, i64 5
  %tmp67 = load i32, i32* %arrayidx8.us.i.5, align 4
  %arrayidx12.us.i61.5 = getelementptr inbounds i32, i32* %pre98, i64 %indvars.iv42.i
  %tmp68 = load i32, i32* %arrayidx12.us.i61.5, align 4
  %mul.us.i.5 = mul nsw i32 %tmp68, %tmp67
  %add.us.i.5 = add nsw i32 %mul.us.i.5, %add.us.i.4
  %arrayidx8.us.i.6 = getelementptr inbounds i32, i32* %tmp56, i64 6
  %tmp69 = load i32, i32* %arrayidx8.us.i.6, align 4
  %arrayidx12.us.i61.6 = getelementptr inbounds i32, i32* %pre99, i64 %indvars.iv42.i
  %tmp70 = load i32, i32* %arrayidx12.us.i61.6, align 4
  %mul.us.i.6 = mul nsw i32 %tmp70, %tmp69
  %add.us.i.6 = add nsw i32 %mul.us.i.6, %add.us.i.5
  %arrayidx8.us.i.7 = getelementptr inbounds i32, i32* %tmp56, i64 7
  %tmp71 = load i32, i32* %arrayidx8.us.i.7, align 4
  %arrayidx12.us.i61.7 = getelementptr inbounds i32, i32* %pre100, i64 %indvars.iv42.i
  %tmp72 = load i32, i32* %arrayidx12.us.i61.7, align 4
  %mul.us.i.7 = mul nsw i32 %tmp72, %tmp71
  %add.us.i.7 = add nsw i32 %mul.us.i.7, %add.us.i.6
  %arrayidx8.us.i.8 = getelementptr inbounds i32, i32* %tmp56, i64 8
  %tmp73 = load i32, i32* %arrayidx8.us.i.8, align 4
  %arrayidx12.us.i61.8 = getelementptr inbounds i32, i32* %pre101, i64 %indvars.iv42.i
  %tmp74 = load i32, i32* %arrayidx12.us.i61.8, align 4
  %mul.us.i.8 = mul nsw i32 %tmp74, %tmp73
  %add.us.i.8 = add nsw i32 %mul.us.i.8, %add.us.i.7
  %arrayidx8.us.i.9 = getelementptr inbounds i32, i32* %tmp56, i64 9
  %tmp75 = load i32, i32* %arrayidx8.us.i.9, align 4
  %arrayidx12.us.i61.9 = getelementptr inbounds i32, i32* %pre102, i64 %indvars.iv42.i
  %tmp76 = load i32, i32* %arrayidx12.us.i61.9, align 4
  %mul.us.i.9 = mul nsw i32 %tmp76, %tmp75
  %add.us.i.9 = add nsw i32 %mul.us.i.9, %add.us.i.8
  %arrayidx16.us.i = getelementptr inbounds i32, i32* %tmp55, i64 %indvars.iv42.i
  store i32 %add.us.i.9, i32* %arrayidx16.us.i, align 4
  %indvars.iv.next43.i = add i64 %indvars.iv42.i, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next43.i to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 10
  br i1 %exitcond, label %end, label %for.body

end:
  ret void
}

; Unlike the above loop, this IR starts out bad and must be
; rescheduled.
;
; CHECK-LABEL: %for.body
; CHECK-NOT: %rsp
; CHECK: imull 4
; CHECK-NOT: {{imull|rsp}}
; CHECK: addl
; CHECK: imull 8
; CHECK-NOT: {{imull|rsp}}
; CHECK: imull 12
; CHECK-NOT: {{imull|rsp}}
; CHECK: addl
; CHECK: addl
; CHECK: imull 16
; CHECK-NOT: {{imull|rsp}}
; CHECK: imull 20
; CHECK-NOT: {{imull|rsp}}
; CHECK: addl
; CHECK: imull 24
; CHECK-NOT: {{imull|rsp}}
; CHECK: addl
; CHECK: addl
; CHECK: imull 28
; CHECK-NOT: {{imull|rsp}}
; CHECK: imull 32
; CHECK-NOT: {{imull|rsp}}
; CHECK: addl
; CHECK: imull 36
; CHECK-NOT: {{imull|rsp}}
; CHECK: addl
; CHECK: addl
; CHECK-NOT: {{imull|rsp}}
; CHECK-LABEL: %end
define void @unrolled_mmult2(i32* %tmp55, i32* %tmp56, i32* %pre, i32* %pre94,
  i32* %pre95, i32* %pre96, i32* %pre97, i32* %pre98, i32* %pre99,
  i32* %pre100, i32* %pre101, i32* %pre102, i32* %pre103, i32* %pre104)
  nounwind uwtable ssp {
entry:
  br label %for.body
for.body:
  %indvars.iv42.i = phi i64 [ %indvars.iv.next43.i, %for.body ], [ 0, %entry ]
  %tmp57 = load i32, i32* %tmp56, align 4
  %arrayidx12.us.i61 = getelementptr inbounds i32, i32* %pre, i64 %indvars.iv42.i
  %tmp58 = load i32, i32* %arrayidx12.us.i61, align 4
  %arrayidx8.us.i.1 = getelementptr inbounds i32, i32* %tmp56, i64 1
  %tmp59 = load i32, i32* %arrayidx8.us.i.1, align 4
  %arrayidx12.us.i61.1 = getelementptr inbounds i32, i32* %pre94, i64 %indvars.iv42.i
  %tmp60 = load i32, i32* %arrayidx12.us.i61.1, align 4
  %arrayidx8.us.i.2 = getelementptr inbounds i32, i32* %tmp56, i64 2
  %tmp61 = load i32, i32* %arrayidx8.us.i.2, align 4
  %arrayidx12.us.i61.2 = getelementptr inbounds i32, i32* %pre95, i64 %indvars.iv42.i
  %tmp62 = load i32, i32* %arrayidx12.us.i61.2, align 4
  %arrayidx8.us.i.3 = getelementptr inbounds i32, i32* %tmp56, i64 3
  %tmp63 = load i32, i32* %arrayidx8.us.i.3, align 4
  %arrayidx12.us.i61.3 = getelementptr inbounds i32, i32* %pre96, i64 %indvars.iv42.i
  %tmp64 = load i32, i32* %arrayidx12.us.i61.3, align 4
  %arrayidx8.us.i.4 = getelementptr inbounds i32, i32* %tmp56, i64 4
  %tmp65 = load i32, i32* %arrayidx8.us.i.4, align 4
  %arrayidx12.us.i61.4 = getelementptr inbounds i32, i32* %pre97, i64 %indvars.iv42.i
  %tmp66 = load i32, i32* %arrayidx12.us.i61.4, align 4
  %arrayidx8.us.i.5 = getelementptr inbounds i32, i32* %tmp56, i64 5
  %tmp67 = load i32, i32* %arrayidx8.us.i.5, align 4
  %arrayidx12.us.i61.5 = getelementptr inbounds i32, i32* %pre98, i64 %indvars.iv42.i
  %tmp68 = load i32, i32* %arrayidx12.us.i61.5, align 4
  %arrayidx8.us.i.6 = getelementptr inbounds i32, i32* %tmp56, i64 6
  %tmp69 = load i32, i32* %arrayidx8.us.i.6, align 4
  %arrayidx12.us.i61.6 = getelementptr inbounds i32, i32* %pre99, i64 %indvars.iv42.i
  %tmp70 = load i32, i32* %arrayidx12.us.i61.6, align 4
  %mul.us.i = mul nsw i32 %tmp58, %tmp57
  %arrayidx8.us.i.7 = getelementptr inbounds i32, i32* %tmp56, i64 7
  %tmp71 = load i32, i32* %arrayidx8.us.i.7, align 4
  %arrayidx12.us.i61.7 = getelementptr inbounds i32, i32* %pre100, i64 %indvars.iv42.i
  %tmp72 = load i32, i32* %arrayidx12.us.i61.7, align 4
  %arrayidx8.us.i.8 = getelementptr inbounds i32, i32* %tmp56, i64 8
  %tmp73 = load i32, i32* %arrayidx8.us.i.8, align 4
  %arrayidx12.us.i61.8 = getelementptr inbounds i32, i32* %pre101, i64 %indvars.iv42.i
  %tmp74 = load i32, i32* %arrayidx12.us.i61.8, align 4
  %arrayidx8.us.i.9 = getelementptr inbounds i32, i32* %tmp56, i64 9
  %tmp75 = load i32, i32* %arrayidx8.us.i.9, align 4
  %arrayidx12.us.i61.9 = getelementptr inbounds i32, i32* %pre102, i64 %indvars.iv42.i
  %tmp76 = load i32, i32* %arrayidx12.us.i61.9, align 4
  %mul.us.i.1 = mul nsw i32 %tmp60, %tmp59
  %add.us.i.1 = add nsw i32 %mul.us.i.1, %mul.us.i
  %mul.us.i.2 = mul nsw i32 %tmp62, %tmp61
  %add.us.i.2 = add nsw i32 %mul.us.i.2, %add.us.i.1
  %mul.us.i.3 = mul nsw i32 %tmp64, %tmp63
  %add.us.i.3 = add nsw i32 %mul.us.i.3, %add.us.i.2
  %mul.us.i.4 = mul nsw i32 %tmp66, %tmp65
  %add.us.i.4 = add nsw i32 %mul.us.i.4, %add.us.i.3
  %mul.us.i.5 = mul nsw i32 %tmp68, %tmp67
  %add.us.i.5 = add nsw i32 %mul.us.i.5, %add.us.i.4
  %mul.us.i.6 = mul nsw i32 %tmp70, %tmp69
  %add.us.i.6 = add nsw i32 %mul.us.i.6, %add.us.i.5
  %mul.us.i.7 = mul nsw i32 %tmp72, %tmp71
  %add.us.i.7 = add nsw i32 %mul.us.i.7, %add.us.i.6
  %mul.us.i.8 = mul nsw i32 %tmp74, %tmp73
  %add.us.i.8 = add nsw i32 %mul.us.i.8, %add.us.i.7
  %mul.us.i.9 = mul nsw i32 %tmp76, %tmp75
  %add.us.i.9 = add nsw i32 %mul.us.i.9, %add.us.i.8
  %arrayidx16.us.i = getelementptr inbounds i32, i32* %tmp55, i64 %indvars.iv42.i
  store i32 %add.us.i.9, i32* %arrayidx16.us.i, align 4
  %indvars.iv.next43.i = add i64 %indvars.iv42.i, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next43.i to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 10
  br i1 %exitcond, label %end, label %for.body

end:
  ret void
}

; A mildly interesting little block extracted from a cipher.  The
; balanced heuristics are interesting here because we have resource,
; latency, and register limits all at once. For now, simply check that
; we don't use any callee-saves.
; CHECK-LABEL: @encpc1
; CHECK-LABEL: %entry
; CHECK-NOT: push
; CHECK-NOT: pop
; CHECK: ret
@a = external global i32, align 4
@b = external global i32, align 4
@c = external global i32, align 4
@d = external global i32, align 4
define i32 @encpc1() nounwind {
entry:
  %l1 = load i32, i32* @a, align 16
  %conv = shl i32 %l1, 8
  %s5 = lshr i32 %l1, 8
  %add = or i32 %conv, %s5
  store i32 %add, i32* @b
  %l6 = load i32, i32* @a
  %l7 = load i32, i32* @c
  %add.i = add i32 %l7, %l6
  %idxprom.i = zext i32 %l7 to i64
  %arrayidx.i = getelementptr inbounds i32, i32* @d, i64 %idxprom.i
  %l8 = load i32, i32* %arrayidx.i
  store i32 346, i32* @c
  store i32 20021, i32* @d
  %l9 = load i32, i32* @a
  store i32 %l8, i32* @a
  store i32 %l9, i32* @b
  store i32 %add.i, i32* @c
  store i32 %l9, i32* @d
  %cmp.i = icmp eq i32 %add.i, 0
  %s10 = lshr i32 %l1, 16
  %s12 = lshr i32 %l1, 24
  %s14 = lshr i32 %l1, 30
  br i1 %cmp.i, label %if, label %return
if:
  %sa = add i32 %s5, %s10
  %sb = add i32 %sa, %s12
  %sc = add i32 %sb, %s14
  br label %return
return:
  %result = phi i32 [0, %entry], [%sc, %if]
  ret i32 %result
}
