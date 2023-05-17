; RUN: opt < %s -passes=indvars -S -o - | FileCheck %s

; When eliminating the overflow intrinsic the indvars pass would incorrectly
; return a false Modified status. This was caught by the pass return
; status check that is hidden under EXPENSIVE_CHECKS.

; CHECK-LABEL: for.body:
; CHECK-NEXT: %0 = phi i16 [ %1, %for.body ], [ undef, %for.body.preheader ]
; CHECK-NEXT: %1 = add nsw i16 %0, -1
; CHECK-NEXT: %cmp = icmp sgt i16 %1, 0
; CHECK-NEXT:  call void @llvm.assume(i1 %cmp)

; Function Attrs: nounwind
define void @foo() #0 {
entry:
  %cmp1 = icmp sgt i16 undef, 0
  br i1 %cmp1, label %for.body.preheader, label %for.end

for.body.preheader:                               ; preds = %entry
  br label %for.body

for.body:                                         ; preds = %for.body.preheader, %for.body
  %0 = phi i16 [ %2, %for.body ], [ undef, %for.body.preheader ]
  %1 = call { i16, i1 } @llvm.sadd.with.overflow.i16(i16 %0, i16 -1)
  %2 = extractvalue { i16, i1 } %1, 0
  %cmp = icmp sgt i16 %2, 0
  call void @llvm.assume(i1 %cmp)
  br label %for.body

for.end:                                          ; preds = %entry
  ret void
}

; Function Attrs: nounwind readnone speculatable willreturn
declare { i16, i1 } @llvm.sadd.with.overflow.i16(i16, i16) #1

; Function Attrs: nounwind willreturn
declare void @llvm.assume(i1) #2

attributes #0 = { nounwind }
attributes #1 = { nounwind readnone speculatable willreturn }
attributes #2 = { nounwind willreturn }

!llvm.ident = !{!0}

!0 = !{!"clang version 12.0.0"}
