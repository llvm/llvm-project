; RUN: opt < %s -passes=loop-vectorize,dce,instcombine -force-vector-interleave=1 -force-vector-width=4 -enable-if-conversion -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

;CHECK-LABEL: @foo(
;CHECK: icmp eq <4 x i32>
;CHECK: select <4 x i1>
;CHECK: ret i32
define i32 @foo(i32 %x, i32 %t, i32* nocapture %A) nounwind uwtable ssp {
entry:
  %cmp10 = icmp sgt i32 %x, 0
  br i1 %cmp10, label %for.body, label %for.end

for.body:                                         ; preds = %entry, %if.end
  %indvars.iv = phi i64 [ %indvars.iv.next, %if.end ], [ 0, %entry ]
  %arrayidx = getelementptr inbounds i32, i32* %A, i64 %indvars.iv
  %0 = load i32, i32* %arrayidx, align 4
  %tobool = icmp eq i32 %0, 0
  br i1 %tobool, label %if.end, label %if.then

if.then:                                          ; preds = %for.body
  %1 = add nsw i64 %indvars.iv, 45
  %2 = trunc i64 %indvars.iv to i32
  %mul = mul nsw i32 %2, %t
  %3 = trunc i64 %1 to i32
  %add1 = add nsw i32 %3, %mul
  br label %if.end

if.end:                                           ; preds = %for.body, %if.then
  %z.0 = phi i32 [ %add1, %if.then ], [ 9, %for.body ]
  store i32 %z.0, i32* %arrayidx, align 4
  %indvars.iv.next = add nsw i64 %indvars.iv, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next to i32
  %exitcond = icmp eq i32 %lftr.wideiv, %x
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %if.end, %entry
  ret i32 undef
}
