; RUN: llc -verify-machineinstrs < %s -mcpu=ppc | FileCheck %s

target datalayout = "E-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v128:128:128-n32"
target triple = "powerpc-unknown-linux-gnu"

define ppc_fp128 @foo(ptr nocapture %n, ppc_fp128 %d) nounwind readonly {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.06 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %x.05 = phi ppc_fp128 [ %d, %entry ], [ %conv, %for.body ]
  %arrayidx = getelementptr inbounds ppc_fp128, ptr %n, i32 %i.06
  %0 = load ppc_fp128, ptr %arrayidx, align 8
  %conv = tail call ppc_fp128 @copysignl(ppc_fp128 %x.05, ppc_fp128 %d) nounwind readonly
  %inc = add nsw i32 %i.06, 1
  %exitcond = icmp eq i32 %inc, 2048
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret ppc_fp128 %conv
}

declare ppc_fp128 @copysignl(ppc_fp128, ppc_fp128) #0

; CHECK: @foo
; CHECK-NOT: mtctr

