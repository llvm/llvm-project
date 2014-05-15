; RUN: llc < %s | FileCheck %s
target datalayout = "E-m:e-p:32:32-i128:64-n32"
target triple = "powerpc-ellcc-linux"

; Function Attrs: nounwind
define void @foo1(i128* %a, i128* readonly %b, i128* readonly %c) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.02 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %0 = load i128* %b, align 16
  %1 = load i128* %c, align 16
  %shl = shl i128 %0, %1
  store i128 %shl, i128* %a, align 16
  %inc = add nsw i32 %i.02, 1
  %exitcond = icmp eq i32 %inc, 2048
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void

; CHECK-LABEL: @foo1
; CHECK-NOT: mtctr
}

; Function Attrs: nounwind
define void @foo2(i128* %a, i128* readonly %b, i128* readonly %c) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.02 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %0 = load i128* %b, align 16
  %1 = load i128* %c, align 16
  %shl = ashr i128 %0, %1
  store i128 %shl, i128* %a, align 16
  %inc = add nsw i32 %i.02, 1
  %exitcond = icmp eq i32 %inc, 2048
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void

; CHECK-LABEL: @foo2
; CHECK-NOT: mtctr
}

; Function Attrs: nounwind
define void @foo3(i128* %a, i128* readonly %b, i128* readonly %c) #0 {
entry:
  br label %for.body

for.body:                                         ; preds = %for.body, %entry
  %i.02 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %0 = load i128* %b, align 16
  %1 = load i128* %c, align 16
  %shl = lshr i128 %0, %1
  store i128 %shl, i128* %a, align 16
  %inc = add nsw i32 %i.02, 1
  %exitcond = icmp eq i32 %inc, 2048
  br i1 %exitcond, label %for.end, label %for.body

for.end:                                          ; preds = %for.body
  ret void

; CHECK-LABEL: @foo3
; CHECK-NOT: mtctr
}

attributes #0 = { nounwind }

