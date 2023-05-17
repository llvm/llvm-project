; RUN: opt < %s -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -S | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

; Function Attrs: nounwind uwtable
define i32 @test1(ptr nocapture %a, ptr nocapture readonly %b) #0 {
entry:
  call void @llvm.pseudoprobe(i64 3666282617048535130, i64 1, i32 0, i64 -1)
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %indvars.iv = phi i64 [ 0, %entry ], [ %indvars.iv.next, %for.body ]
  %arrayidx = getelementptr inbounds float, ptr %b, i64 %indvars.iv
  %0 = load float, ptr %arrayidx, align 4
  %conv = fptosi float %0 to i32
  %arrayidx2 = getelementptr inbounds i32, ptr %a, i64 %indvars.iv
  store i32 %conv, ptr %arrayidx2, align 4
  %indvars.iv.next = add nuw nsw i64 %indvars.iv, 1
  %exitcond = icmp eq i64 %indvars.iv.next, 1600
  call void @llvm.pseudoprobe(i64 3666282617048535130, i64 2, i32 0, i64 -1)
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  call void @llvm.pseudoprobe(i64 3666282617048535130, i64 3, i32 0, i64 -1)
  ret i32 0
}


; CHECK-LABEL:  @test1
; CHECK:        vector.body:
; CHECK:          load <4 x float>, ptr %{{.*}}
; CHECK:          store <4 x i32> %{{.*}}, ptr %{{.*}}
; CHECK-COUNT-4:  call void @llvm.pseudoprobe(i64 3666282617048535130, i64 2, i32 0, i64 -1)
; CHECK:          %index.next = add nuw i64 %index, 4




; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.pseudoprobe(i64, i64, i32, i64) #1

attributes #0 = { nounwind uwtable }
attributes #1 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.pseudo_probe_desc = !{!0}

!0 = !{i64 3666282617048535130, i64 52824598631, !"test1"}
