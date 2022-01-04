; RUN: llc < %s -mcpu=a2 -verify-machineinstrs | FileCheck %s
; RUN: llc < %s -mcpu=a2 -disable-lsr -verify-machineinstrs | FileCheck -check-prefix=NOLSR %s
target datalayout = "E-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-f128:128:128-v128:128:128-n32:64"
target triple = "powerpc64-unknown-linux-gnu"

define void @main() #0 {
entry:
  br i1 undef, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 1, %entry ]
  %indvars.iv.next = add i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, 0
  br i1 %exitcond, label %for.end, label %for.body

; CHECK: @main
; CHECK: li [[REG:[0-9]+]], -1
; CHECK: rldic [[REG2:[0-9]+]], [[REG]], 0, 32
; CHECK: mtctr [[REG2]]
; CHECK: bdnz

for.end:                                          ; preds = %for.body, %entry
  ret void
}

define void @main1() #0 {
entry:
  br i1 undef, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 1, %entry ]
  %indvars.iv.next = add i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 0
  br i1 %exitcond, label %for.end, label %for.body

; CHECK: @main1
; CHECK: li [[REG:[0-9]+]], -1
; CHECK: mtctr [[REG]]
; CHECK: bdnz

for.end:                                          ; preds = %for.body, %entry
  ret void
}

define void @main2() #0 {
entry:
  br i1 undef, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 1, %entry ]
  %indvars.iv.next = add i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, -100000
  br i1 %exitcond, label %for.end, label %for.body

; CHECK: @main2
; CHECK: lis [[REG:[0-9]+]], -2
; CHECK: ori [[REG2:[0-9]+]], [[REG]], 31071
; CHECK: mtctr [[REG2]]
; CHECK: bdnz

for.end:                                          ; preds = %for.body, %entry
  ret void
}

define void @main3() #0 {
entry:
  br i1 undef, label %for.end, label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ %indvars.iv.next, %for.body ], [ 127984, %entry ]
  %indvars.iv.next = add i64 %indvars.iv, -16
  %exitcond = icmp eq i64 %indvars.iv.next, -16
  br i1 %exitcond, label %for.end, label %for.body

; NOLSR: @main3
; NOLSR: li [[REG:[0-9]+]], 8000
; NOLSR: mtctr [[REG]]
; NOLSR: bdnz

for.end:                                          ; preds = %for.body, %entry
  ret void
}

attributes #0 = { nounwind "less-precise-fpmad"="false" "frame-pointer"="non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "unsafe-fp-math"="false" "use-soft-float"="false" }
