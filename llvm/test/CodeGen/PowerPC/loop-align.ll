; Test the loop alignment.
; RUN: llc -verify-machineinstrs -mcpu=a2 -mtriple powerpc64le-unknown-linux-gnu < %s | FileCheck %s -check-prefixes=CHECK,GENERIC
; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mtriple powerpc64le-unknown-linux-gnu < %s | FileCheck %s -check-prefixes=CHECK,PWR
; RUN: llc -verify-machineinstrs -mcpu=pwr9 -mtriple powerpc64le-unknown-linux-gnu < %s | FileCheck %s -check-prefixes=CHECK,PWR
; RUN: llc -verify-machineinstrs -mcpu=pwr8 -mtriple powerpc64-unknown-linux-gnu < %s | FileCheck %s -check-prefixes=CHECK,PWR
; RUN: llc -verify-machineinstrs -mcpu=pwr9 -mtriple powerpc64-unknown-linux-gnu < %s | FileCheck %s -check-prefixes=CHECK,PWR

; Test the loop alignment and the option -disable-ppc-innermost-loop-align32.
; RUN: llc -verify-machineinstrs -mcpu=a2 -disable-ppc-innermost-loop-align32 -mtriple powerpc64le-unknown-linux-gnu < %s | FileCheck %s -check-prefixes=CHECK,GENERIC-DISABLE-PPC-INNERMOST-LOOP-ALIGN32
; RUN: llc -verify-machineinstrs -mcpu=pwr8 -disable-ppc-innermost-loop-align32 -mtriple powerpc64le-unknown-linux-gnu < %s | FileCheck %s -check-prefixes=CHECK,PWR-DISABLE-PPC-INNERMOST-LOOP-ALIGN32
; RUN: llc -verify-machineinstrs -mcpu=pwr9 -disable-ppc-innermost-loop-align32 -mtriple powerpc64le-unknown-linux-gnu < %s | FileCheck %s -check-prefixes=CHECK,PWR-DISABLE-PPC-INNERMOST-LOOP-ALIGN32
; RUN: llc -verify-machineinstrs -mcpu=pwr8 -disable-ppc-innermost-loop-align32 -mtriple powerpc64-unknown-linux-gnu < %s | FileCheck %s -check-prefixes=CHECK,PWR-DISABLE-PPC-INNERMOST-LOOP-ALIGN32
; RUN: llc -verify-machineinstrs -mcpu=pwr9 -disable-ppc-innermost-loop-align32 -mtriple powerpc64-unknown-linux-gnu < %s | FileCheck %s -check-prefixes=CHECK,PWR-DISABLE-PPC-INNERMOST-LOOP-ALIGN32


%struct.parm = type { ptr, i32, i32 }

; Test the loop alignment when the innermost hot loop has more than 8 instructions.
define void @big_loop(ptr %arg) {
entry:
  %localArg.sroa.0.0.copyload = load ptr, ptr %arg, align 8
  %localArg.sroa.4.0..sroa_idx56 = getelementptr inbounds %struct.parm, ptr %arg, i64 0, i32 1
  %localArg.sroa.4.0.copyload = load i32, ptr %localArg.sroa.4.0..sroa_idx56, align 8
  %localArg.sroa.5.0..sroa_idx58 = getelementptr inbounds %struct.parm, ptr %arg, i64 0, i32 2
  %localArg.sroa.5.0.copyload = load i32, ptr %localArg.sroa.5.0..sroa_idx58, align 4
  %0 = sext i32 %localArg.sroa.5.0.copyload to i64
  br label %do.body

do.body:                                          ; preds = %do.end, %entry
  %m.0 = phi i32 [ %localArg.sroa.4.0.copyload, %entry ], [ %dec24, %do.end ]
  br label %do.body3

do.body3:                                         ; preds = %do.body3, %do.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %do.body3 ], [ %0, %do.body ]
  %1 = add nsw i64 %indvars.iv, 2
  %arrayidx = getelementptr inbounds i32, ptr %localArg.sroa.0.0.copyload, i64 %1
  %2 = add nsw i64 %indvars.iv, 3
  %3 = trunc i64 %1 to i32
  %4 = add nsw i64 %indvars.iv, 4
  %arrayidx10 = getelementptr inbounds i32, ptr %localArg.sroa.0.0.copyload, i64 %2
  %5 = trunc i64 %2 to i32
  store i32 %5, ptr %arrayidx10, align 4
  %arrayidx12 = getelementptr inbounds i32, ptr %localArg.sroa.0.0.copyload, i64 %4
  %6 = trunc i64 %4 to i32
  store i32 %6, ptr %arrayidx12, align 4
  store i32 %3, ptr %arrayidx, align 4
  %arrayidx21 = getelementptr inbounds i32, ptr %localArg.sroa.0.0.copyload, i64 %indvars.iv
  %7 = trunc i64 %indvars.iv to i32
  %8 = add i32 %7, 1
  store i32 %8, ptr %arrayidx21, align 4
  %indvars.iv.next = add nsw i64 %indvars.iv, -1
  %9 = icmp eq i64 %indvars.iv, 0
  br i1 %9, label %do.end, label %do.body3

do.end:                                           ; preds = %do.body3
  %dec24 = add nsw i32 %m.0, -1
  %tobool25 = icmp eq i32 %m.0, 0
  br i1 %tobool25, label %do.end26, label %do.body

do.end26:                                         ; preds = %do.end
  %arrayidx28 = getelementptr inbounds i32, ptr %localArg.sroa.0.0.copyload, i64 %0
  store i32 0, ptr %arrayidx28, align 4
  ret void


; CHECK-LABEL: @big_loop
; CHECK: mtctr 
; GENERIC: .p2align  4
; PWR: .p2align  5
; GENERIC-DISABLE-PPC-INNERMOST-LOOP-ALIGN32: .p2align  4
; PWR-DISABLE-PPC-INNERMOST-LOOP-ALIGN32: .p2align  4  
; CHECK: bdnz 
}

; Test the loop alignment when the innermost hot loop has 5-8 instructions.
define void @general_loop(ptr %s, i64 %m) {
entry:
  %tobool40 = icmp eq i64 %m, 0
  br i1 %tobool40, label %while.end18, label %while.body3.lr.ph

while.cond.loopexit:                              ; preds = %while.body3
  %tobool = icmp eq i64 %dec, 0
  br i1 %tobool, label %while.end18, label %while.body3.lr.ph

while.body3.lr.ph:                                ; preds = %entry, %while.cond.loopexit
  %m.addr.041 = phi i64 [ %dec, %while.cond.loopexit ], [ %m, %entry ]
  %dec = add nsw i64 %m.addr.041, -1
  %conv = trunc i64 %m.addr.041 to i32
  %conv11 = trunc i64 %dec to i32
  br label %while.body3

while.body3:                                      ; preds = %while.body3.lr.ph, %while.body3
  %n.039 = phi i64 [ %m.addr.041, %while.body3.lr.ph ], [ %dec16, %while.body3 ]
  %inc = add nsw i64 %n.039, 1
  %arrayidx = getelementptr inbounds i32, ptr %s, i64 %n.039
  %inc5 = add nsw i64 %n.039, 2
  %arrayidx6 = getelementptr inbounds i32, ptr %s, i64 %inc
  %sub = sub nsw i64 %dec, %inc5
  %conv7 = trunc i64 %sub to i32
  %arrayidx9 = getelementptr inbounds i32, ptr %s, i64 %inc5
  store i32 %conv7, ptr %arrayidx9, align 4
  store i32 %conv11, ptr %arrayidx6, align 4
  store i32 %conv, ptr %arrayidx, align 4
  %dec16 = add nsw i64 %n.039, -1
  %tobool2 = icmp eq i64 %dec16, 0
  br i1 %tobool2, label %while.cond.loopexit, label %while.body3

while.end18:                                      ; preds = %while.cond.loopexit, %entry
  ret void


; CHECK-LABEL: @general_loop
; CHECK: mtctr 
; GENERIC: .p2align  4
; PWR: .p2align  5
; GENERIC-DISABLE-PPC-INNERMOST-LOOP-ALIGN32: .p2align  4
; PWR-DISABLE-PPC-INNERMOST-LOOP-ALIGN32: .p2align  5  
; CHECK: bdnz
}

; Test the small loop alignment when the innermost hot loop has less than 4 instructions.
define void @small_loop(i64 %m) {
entry:
  br label %do.body

do.body:                                          ; preds = %do.end, %entry
  %m.addr.0 = phi i64 [ %m, %entry ], [ %1, %do.end ]
  br label %do.body1

do.body1:                                         ; preds = %do.body1, %do.body
  %n.0 = phi i64 [ %m.addr.0, %do.body ], [ %0, %do.body1 ]
  %0 = tail call i64 asm "subi     $0,$0,1", "=r,0"(i64 %n.0)
  %tobool = icmp eq i64 %0, 0
  br i1 %tobool, label %do.end, label %do.body1

do.end:                                           ; preds = %do.body1
  %1 = tail call i64 asm "subi     $1,$1,1", "=r,0"(i64 %m.addr.0)
  %tobool3 = icmp eq i64 %1, 0
  br i1 %tobool3, label %do.end4, label %do.body

do.end4:                                          ; preds = %do.end
  ret void


; CHECK-LABEL: @small_loop
; CHECK: mr 
; GENERIC: .p2align  4
; PWR: .p2align  5
; GENERIC-DISABLE-PPC-INNERMOST-LOOP-ALIGN32: .p2align  4
; PWR-DISABLE-PPC-INNERMOST-LOOP-ALIGN32: .p2align  4  
; CHECK: bne
}

; Test the loop alignment when the innermost cold loop has more than 8 instructions.
define void @big_loop_cold_innerloop(ptr %arg) {
entry:
  %localArg.sroa.0.0.copyload = load ptr, ptr %arg, align 8
  %localArg.sroa.4.0..sroa_idx56 = getelementptr inbounds %struct.parm, ptr %arg, i64 0, i32 1
  %localArg.sroa.4.0.copyload = load i32, ptr %localArg.sroa.4.0..sroa_idx56, align 8
  %localArg.sroa.5.0..sroa_idx58 = getelementptr inbounds %struct.parm, ptr %arg, i64 0, i32 2
  %localArg.sroa.5.0.copyload = load i32, ptr %localArg.sroa.5.0..sroa_idx58, align 4
  %0 = sext i32 %localArg.sroa.5.0.copyload to i64
  br label %do.body

do.body:                                          ; preds = %do.end, %entry
  %m.0 = phi i32 [ %localArg.sroa.4.0.copyload, %entry ], [ %dec24, %do.end ]
  br label %do.body3

do.body3:                                         ; preds = %do.body3, %do.body
  %indvars.iv = phi i64 [ %indvars.iv.next, %do.body3 ], [ %0, %do.body ]
  %1 = add nsw i64 %indvars.iv, 2
  %arrayidx = getelementptr inbounds i32, ptr %localArg.sroa.0.0.copyload, i64 %1
  %2 = add nsw i64 %indvars.iv, 3
  %3 = trunc i64 %1 to i32
  %4 = add nsw i64 %indvars.iv, 4
  %arrayidx10 = getelementptr inbounds i32, ptr %localArg.sroa.0.0.copyload, i64 %2
  %5 = trunc i64 %2 to i32
  store i32 %5, ptr %arrayidx10, align 4
  %arrayidx12 = getelementptr inbounds i32, ptr %localArg.sroa.0.0.copyload, i64 %4
  %6 = trunc i64 %4 to i32
  store i32 %6, ptr %arrayidx12, align 4
  store i32 %3, ptr %arrayidx, align 4
  %arrayidx21 = getelementptr inbounds i32, ptr %localArg.sroa.0.0.copyload, i64 %indvars.iv
  %7 = trunc i64 %indvars.iv to i32
  %8 = add i32 %7, 1
  store i32 %8, ptr %arrayidx21, align 4
  %indvars.iv.next = add nsw i64 %indvars.iv, -1
  %9 = icmp eq i64 %indvars.iv, 0
  br i1 %9, label %do.end, label %do.body3

do.end:                                           ; preds = %do.body3
  %dec24 = add nsw i32 %m.0, -1
  %tobool25 = icmp eq i32 %m.0, 0
  br i1 %tobool25, label %do.end26, label %do.body

do.end26:                                         ; preds = %do.end
  %arrayidx28 = getelementptr inbounds i32, ptr %localArg.sroa.0.0.copyload, i64 %0
  store i32 0, ptr %arrayidx28, align 4
  ret void


; CHECK-LABEL: @big_loop_cold_innerloop
; CHECK: mtctr 
; PWR: .p2align 5
; CHECK-NOT: .p2align 5
; CHECK: bdnz 
}
