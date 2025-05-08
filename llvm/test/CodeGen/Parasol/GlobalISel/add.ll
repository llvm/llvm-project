; Modified by Sunscreen under the AGPLv3 license; see the README at the
; repository root for more information
;
; ModuleID = 'add.bc'
source_filename = "add.c"
target datalayout = "e-m:e-p:32:32-i64:64-n1:8:16:32-S128"
target triple = "parasol"

; Function Attrs: fhe_circuit mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite)
define dso_local void @add(ptr encrypted nocapture noundef readonly %a, ptr encrypted nocapture noundef readonly %b, ptr encrypted nocapture noundef writeonly %output) local_unnamed_addr #0 {
entry:
  %0 = load i8, ptr %a, align 1, !tbaa !3
  %1 = load i8, ptr %b, align 1, !tbaa !3
  %add = add i8 %1, %0
  store i8 %add, ptr %output, align 1, !tbaa !3
  ret void
}

attributes #0 = { fhe_circuit mustprogress nofree norecurse nosync nounwind willreturn memory(argmem: readwrite) "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }

!llvm.module.flags = !{!0, !1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{!"clang version 18.1.6 (git@github.com:Sunscreen-tech/tfhe-llvm.git 362532c3748e80797f68ea9d922a565dd1cd2465)"}
!3 = !{!4, !4, i64 0}
!4 = !{!"omnipotent char", !5, i64 0}
!5 = !{!"Simple C/C++ TBAA"}

