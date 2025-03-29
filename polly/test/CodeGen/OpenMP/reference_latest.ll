; RUN: opt %loadNPMPolly '-passes=polly-delicm,polly-simplify,polly-codegen' -polly-parallel -S < %s | FileCheck %s
;
; Test that parallel codegen handles scalars mapped to other arrays.
; After mapping "store double %add10" references the array "MemRef2".
; Its base pointer therefore needs to be made available in the subfunction.

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"

define void @reference_latest(ptr nocapture readonly %data, i32 %n, i32 %m) {
entry:
  %0 = alloca double, i64 undef, align 16
  %conv1 = sext i32 %m to i64
  br label %while.body

while.body:
  %indvars.iv211 = phi i64 [ %conv1, %entry ], [ %indvars.iv.next212, %for.end ]
  br label %for.body

for.body:
  %indvars.iv207 = phi i64 [ %indvars.iv211, %while.body ], [ %indvars.iv.next208, %for.body ]
  %1 = load float, ptr %data, align 4
  %add10 = fadd double undef, undef
  %indvars.iv.next208 = add nsw i64 %indvars.iv207, 1
  %lftr.wideiv = trunc i64 %indvars.iv.next208 to i32
  %exitcond210 = icmp eq i32 %lftr.wideiv, %n
  br i1 %exitcond210, label %for.end, label %for.body

for.end:
  %arrayidx12 = getelementptr inbounds double, ptr %0, i64 %indvars.iv211
  store double %add10, ptr %arrayidx12, align 8
  %indvars.iv.next212 = add nsw i64 %indvars.iv211, -1
  %2 = trunc i64 %indvars.iv211 to i32
  %tobool = icmp eq i32 %2, 0
  br i1 %tobool, label %while.end, label %while.body

while.end:
  ret void
}

; CHECK-LABEL: define internal void @reference_latest_polly_subfn(ptr %polly.par.userContext)

; CHECK:      %polly.access.polly.subfunc.arg. = getelementptr double, ptr %polly.subfunc.arg., i64 %{{[0-9]+}}
; CHECK-NEXT: store double %p_add{{[0-9]*}}, ptr %polly.access.polly.subfunc.arg.
