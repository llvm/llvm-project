; RUN: opt -S -codegenprepare -disable-complex-addr-modes=false -addr-sink-new-phis=true -addr-sink-new-select=true  %s | FileCheck %s

target triple = "x86_64-unknown-linux-gnu"
target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v64:64:64-v128:128:128-a0:0:64-s0:64:64-f80:128:128"

@gv = global [2 x i16] undef

define void @f2() {
entry:
  %arrayidx1 = getelementptr inbounds [2 x i16], ptr @gv, i16 0, i16 1
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %e.03 = phi ptr [ @gv, %entry ], [ %arrayidx1, %for.body ]
  %tobool = icmp eq i16 undef, 0
  br i1 %tobool, label %for.body, label %for.end

for.end:                                          ; preds = %for.body
; CHECK: sunkaddr
  %e.1.le = select i1 %tobool, ptr %arrayidx1, ptr %e.03
  store i16 0, ptr %e.1.le, align 1
  ret void
}
