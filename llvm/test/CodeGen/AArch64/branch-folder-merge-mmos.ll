; RUN: llc < %s -mtriple=aarch64-none-linux-gnu -stop-after branch-folder | FileCheck %s
target datalayout = "e-m:e-i64:64-i128:128-n32:64-S128"

; Function Attrs: norecurse nounwind
define void @foo(i32 %a, i32 %b, float* nocapture %foo_arr) #0 {
; CHECK: (load 4 from %ir.arrayidx1.{{i[1-2]}}), (load 4 from %ir.arrayidx1.{{i[1-2]}})
entry:
  %cmp = icmp sgt i32 %a, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  %0 = load float, float* %foo_arr, align 4
  %arrayidx1.i1 = getelementptr inbounds float, float* %foo_arr, i64 1
  %1 = load float, float* %arrayidx1.i1, align 4
  %sub.i = fsub float %0, %1
  store float %sub.i, float* %foo_arr, align 4
  br label %if.end3

if.end:                                           ; preds = %entry
  %cmp1 = icmp sgt i32 %b, 0
  br i1 %cmp1, label %if.then2, label %if.end3

if.then2:                                         ; preds = %if.end
  %2 = load float, float* %foo_arr, align 4
  %arrayidx1.i2 = getelementptr inbounds float, float* %foo_arr, i64 1
  %3 = load float, float* %arrayidx1.i2, align 4
  %sub.i3 = fsub float %2, %3
  store float %sub.i3, float* %foo_arr, align 4
  br label %if.end3

if.end3:                                          ; preds = %if.then2, %if.end, %if.then
  ret void
}
