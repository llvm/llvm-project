; RUN: llc < %s -mtriple=thumbv7-apple-darwin -mcpu=cortex-a8 | FileCheck %s

; Overly aggressive LICM simply adds copies of constants
; rdar://9266679

define zeroext i1 @t(i32* nocapture %A, i32 %size, i32 %value) nounwind readonly ssp {
; CHECK: t:
entry:
  br label %for.cond

for.cond:
  %0 = phi i32 [ 0, %entry ], [ %inc, %for.inc ]
  %cmp = icmp ult i32 %0, %size
  br i1 %cmp, label %for.body, label %return

for.body:
; CHECK: %for.
; CHECK: movs r{{[0-9]+}}, #{{[01]}}
  %arrayidx = getelementptr i32* %A, i32 %0
  %tmp4 = load i32* %arrayidx, align 4
  %cmp6 = icmp eq i32 %tmp4, %value
  br i1 %cmp6, label %return, label %for.inc

; CHECK: %for.
; CHECK: movs r{{[0-9]+}}, #{{[01]}}

for.inc:
  %inc = add i32 %0, 1
  br label %for.cond

return:
  %retval.0 = phi i1 [ true, %for.body ], [ false, %for.cond ]
  ret i1 %retval.0
}
