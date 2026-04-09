; RUN: opt < %s -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=1 -S | FileCheck %s --check-prefix=VF1UF1
; RUN: opt < %s -passes=loop-vectorize -force-vector-interleave=2 -force-vector-width=1 -S | FileCheck %s --check-prefix=VF1UF2
; RUN: opt < %s -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=2 -S | FileCheck %s --check-prefix=VF2UF1
; RUN: opt < %s -passes=loop-vectorize -force-vector-interleave=1 -force-vector-width=4 -S | FileCheck %s --check-prefix=VF4UF1
; RUN: opt < %s -passes=loop-vectorize -force-vector-interleave=2 -force-vector-width=4 -S | FileCheck %s --check-prefix=VF4UF2

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define i32 @test1(ptr nocapture %a, ptr nocapture readonly %b) {
; VF1UF1-LABEL:  @test1
; VF1UF1:        for.body:
; VF1UF1-COUNT-1:  call void @llvm.pseudoprobe
; VF1UF1-NOT:      call void @llvm.pseudoprobe
; VF1UF1:          br i1
;
; VF1UF2-LABEL:  @test1
; VF1UF2:        vector.body:
; VF1UF2-COUNT-2:  call void @llvm.pseudoprobe
; VF1UF2-NOT:      call void @llvm.pseudoprobe
; VF1UF2:          %index.next = add nuw i64 %index, 2
;
; VF2UF1-LABEL:  @test1
; VF2UF1:        vector.body:
; VF2UF1:          load <2 x float>, ptr %{{.*}}
; VF2UF1:          store <2 x i32> %{{.*}}, ptr %{{.*}}
; VF2UF1-COUNT-2:  call void @llvm.pseudoprobe
; VF2UF1-NOT:      call void @llvm.pseudoprobe
; VF2UF1:          %index.next = add nuw i64 %index, 2
;
; VF4UF1-LABEL:  @test1
; VF4UF1:        vector.body:
; VF4UF1:          load <4 x float>, ptr %{{.*}}
; VF4UF1:          store <4 x i32> %{{.*}}, ptr %{{.*}}
; VF4UF1-COUNT-4:  call void @llvm.pseudoprobe
; VF4UF1-NOT:      call void @llvm.pseudoprobe
; VF4UF1:          %index.next = add nuw i64 %index, 4
;
; VF4UF2-LABEL:  @test1
; VF4UF2:        vector.body:
; VF4UF2:          load <4 x float>, ptr %{{.*}}
; VF4UF2:          store <4 x i32> %{{.*}}, ptr %{{.*}}
; VF4UF2-COUNT-8:  call void @llvm.pseudoprobe
; VF4UF2-NOT:      call void @llvm.pseudoprobe
; VF4UF2:          %index.next = add nuw i64 %index, 8
;
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


!llvm.pseudo_probe_desc = !{!0}

!0 = !{i64 3666282617048535130, i64 52824598631, !"test1"}
