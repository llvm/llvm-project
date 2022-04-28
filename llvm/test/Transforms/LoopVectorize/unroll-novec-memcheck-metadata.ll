; RUN: opt < %s -passes=loop-vectorize -force-vector-interleave=2 -force-vector-width=1 -S | FileCheck --enable-var-scope %s

; Make sure we attach memcheck metadata to scalarized memory operations even if
; we're only unrolling.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; CHECK-LABEL: vector.memcheck:
; CHECK-LABEL: vector.body:
; CHECK: load i32, {{.*}} !alias.scope ![[$MD1:[0-9]+]]
; CHECK-LABEL: middle.block:
; CHECK-DAG: ![[$MD1]] = !{![[MD2:[0-9]+]]}
; CHECK-DAG: ![[MD2]] = distinct !{![[MD2]], ![[MD3:[0-9]+]]}
; CHECK-DAG: ![[MD3]] = distinct !{![[MD3]], !"LVerDomain"}

; Function Attrs: norecurse nounwind uwtable
define void @test(i32* nocapture readonly %a, i32* nocapture %b) local_unnamed_addr #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds i32, i32* %a, i64 %indvars.iv
  %l.1 = load i32, i32* %arrayidx, align 4
  %arrayidx2 = getelementptr inbounds i32, i32* %b, i64 %indvars.iv
  %l.2 = load i32, i32* %arrayidx2
  %add = add nsw i32 %l.1, %l.2
  store i32 %add, i32* %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 10000
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void
}

attributes #0 = { norecurse nounwind uwtable }
