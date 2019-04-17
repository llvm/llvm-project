; RUN: opt -S -indvars < %s | FileCheck %s

; PR12627
define void @test1(i32 %x) nounwind uwtable ssp {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %phi1 = phi i1 [ false, %entry ], [ %cmpa, %for.body ]
  %phi2 = phi i1 [ false, %entry ], [ %cmpb, %for.body ]
  %i.07 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  tail call void @aux(i1 %phi1, i1 %phi2) nounwind
  %cmpa = icmp sgt i32 %i.07, 200
  %cmpb = icmp sgt i32 %i.07, 100
  %inc = add nsw i32 %i.07, 1
  %exitcond = icmp eq i32 %inc, 100
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void

; CHECK-LABEL: @test1(
; CHECK-NOT: phi i1
; CHECK: call void @aux(i1 false, i1 false)
}

declare void @aux(i1, i1)
