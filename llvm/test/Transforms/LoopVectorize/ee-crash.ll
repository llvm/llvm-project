; RUN: opt < %s -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -S | FileCheck %s

target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128-n8:16:32:64-S128"

; This test checks that we deal with an in-loop extractelement (for now, this
; means not crashing by not vectorizing).
; CHECK-LABEL: @_Z4foo1Pii(
; CHECK-NOT: <4 x i32>
; CHECK: ret
define i32 @_Z4foo1Pii(ptr %A, i32 %n, <2 x i32> %q) #0 {
entry:
  %idx.ext = sext i32 %n to i64
  %add.ptr = getelementptr inbounds i32, ptr %A, i64 %idx.ext
  %cmp3.i = icmp eq i32 %n, 0
  br i1 %cmp3.i, label %_ZSt10accumulateIPiiET0_T_S2_S1_.exit, label %for.body.i

for.body.i:                                       ; preds = %entry, %for.body.i
  %__init.addr.05.i = phi i32 [ %add.i, %for.body.i ], [ 0, %entry ]
  %__first.addr.04.i = phi ptr [ %incdec.ptr.i, %for.body.i ], [ %A, %entry ]
  %0 = load i32, ptr %__first.addr.04.i, align 4
  %q1 = extractelement <2 x i32> %q, i32 %n
  %q2 = add nsw i32 %0, %q1
  %add.i = add nsw i32 %q2, %__init.addr.05.i
  %incdec.ptr.i = getelementptr inbounds i32, ptr %__first.addr.04.i, i64 1
  %cmp.i = icmp eq ptr %incdec.ptr.i, %add.ptr
  br i1 %cmp.i, label %_ZSt10accumulateIPiiET0_T_S2_S1_.exit, label %for.body.i

_ZSt10accumulateIPiiET0_T_S2_S1_.exit:            ; preds = %for.body.i, %entry
  %__init.addr.0.lcssa.i = phi i32 [ 0, %entry ], [ %add.i, %for.body.i ]
  ret i32 %__init.addr.0.lcssa.i
}

attributes #0 = { nounwind readonly ssp uwtable }

