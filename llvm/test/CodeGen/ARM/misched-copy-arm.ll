; REQUIRES: asserts
; RUN: llc -mtriple=thumb-eabi -mcpu=swift -pre-RA-sched=source -join-globalcopies -enable-misched -verify-misched -debug-only=machine-scheduler -arm-atomic-cfg-tidy=0 %s -o - 2>&1 | FileCheck %s
;
; Loop counter copies should be eliminated.
; There is also a MUL here, but we don't care where it is scheduled.
; CHECK: postinc
; CHECK: *** Final schedule for %bb.2 ***
; CHECK: t2LDRs
; CHECK: t2ADDrr
; CHECK: t2CMPrr
; CHECK: COPY
define i32 @postinc(i32 %a, ptr nocapture %d, i32 %s) nounwind {
entry:
  %cmp4 = icmp eq i32 %a, 0
  br i1 %cmp4, label %for.end, label %for.body

for.body:                                         ; preds = %entry, %for.body
  %indvars.iv = phi i32 [ %indvars.iv.next, %for.body ], [ 0, %entry ]
  %s.05 = phi i32 [ %mul, %for.body ], [ 0, %entry ]
  %indvars.iv.next = add i32 %indvars.iv, %s
  %arrayidx = getelementptr inbounds i32, ptr %d, i32 %indvars.iv
  %0 = load i32, ptr %arrayidx, align 4
  %mul = mul nsw i32 %0, %s.05
  %exitcond = icmp eq i32 %indvars.iv.next, %a
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body, %entry
  %s.0.lcssa = phi i32 [ 0, %entry ], [ %mul, %for.body ]
  ret i32 %s.0.lcssa
}


; This case was a crasher in constrainLocalCopy.
; The problem was the t2LDR_PRE defining both the global and local lrg.
; CHECK-LABEL: *** Final schedule for %bb.5 ***
; CHECK: %[[R4:[0-9]+]]:gpr, %[[R1:[0-9]+]]:gpr = t2LDR_PRE %[[R1]]
; CHECK: %{{[0-9]+}}:gpr = COPY %[[R1]]
; CHECK: %{{[0-9]+}}:gpr = COPY %[[R4]]
; CHECK-LABEL: MACHINEINSTRS
%struct.rtx_def = type { [4 x i8], [1 x %union.rtunion_def] }
%union.rtunion_def = type { i64 }

; Function Attrs: nounwind ssp
declare hidden fastcc void @df_ref_record(ptr nocapture, ptr, ptr, ptr, i32, i32) #0

; Function Attrs: nounwind ssp
define hidden fastcc void @df_def_record_1(ptr nocapture %df, ptr %x, ptr %insn) #0 {
entry:
  br label %while.cond

while.cond:                                       ; preds = %if.end28, %entry
  %loc.0 = phi ptr [ %rtx31, %if.end28 ], [ undef, %entry ]
  %dst.0 = phi ptr [ %0, %if.end28 ], [ undef, %entry ]
  switch i32 undef, label %if.end47 [
    i32 61, label %if.then46
    i32 64, label %if.then24
    i32 132, label %if.end28
    i32 133, label %if.end28
  ]

if.then24:                                        ; preds = %while.cond
  br label %if.end28

if.end28:                                         ; preds = %if.then24, %while.cond, %while.cond
  %dst.1 = phi ptr [ undef, %if.then24 ], [ %dst.0, %while.cond ], [ %dst.0, %while.cond ]
  %arrayidx30 = getelementptr inbounds %struct.rtx_def, ptr %dst.1, i32 0, i32 1, i32 0
  %rtx31 = bitcast ptr %arrayidx30 to ptr
  %0 = load ptr, ptr %rtx31, align 4
  br label %while.cond

if.then46:                                        ; preds = %while.cond
  tail call fastcc void @df_ref_record(ptr %df, ptr %dst.0, ptr %loc.0, ptr %insn, i32 0, i32 undef)
  unreachable

if.end47:                                         ; preds = %while.cond
  ret void
}

attributes #0 = { nounwind ssp }
