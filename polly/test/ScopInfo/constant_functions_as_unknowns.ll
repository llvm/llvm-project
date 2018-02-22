; RUN: opt %loadPolly -polly-scops -analyze < %s | FileCheck %s

target datalayout = "e-m:e-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; CHECK:      Context:
; CHECK-NEXT: [__global_id_0] -> {  : -9223372036854775808 <= __global_id_0 <= 9223372036854775807 }
; CHECK-NEXT: Assumed Context:
; CHECK-NEXT: [__global_id_0] -> {  :  }
; CHECK-NEXT: Invalid Context:
; CHECK-NEXT: [__global_id_0] -> {  : false }
; CHECK-NEXT: p0: %__global_id_0
; CHECK-NEXT: Arrays {
; CHECK-NEXT:     i64 MemRef_A[*]; // Element size 8
; CHECK-NEXT: }
; CHECK-NEXT: Arrays (Bounds as pw_affs) {
; CHECK-NEXT:     i64 MemRef_A[*]; // Element size 8
; CHECK-NEXT: }
; CHECK-NEXT: Alias Groups (0):
; CHECK-NEXT:     n/a
; CHECK-NEXT: Statements {
; CHECK-NEXT: 	Stmt_bb
; CHECK-NEXT:         Domain :=
; CHECK-NEXT:             [__global_id_0] -> { Stmt_bb[] };
; CHECK-NEXT:         Schedule :=
; CHECK-NEXT:             [__global_id_0] -> { Stmt_bb[] -> [] };
; CHECK-NEXT:         MustWriteAccess :=	[Reduction Type: NONE] [Scalar: 0]
; CHECK-NEXT:             [__global_id_0] -> { Stmt_bb[] -> MemRef_A[__global_id_0] };
; CHECK-NEXT: }

define void @globalid(i64* nocapture %A) local_unnamed_addr #0 !kernel_arg_addr_space !2 !kernel_arg_access_qual !3 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !5 {
entry:
  br label %next

next:
  br i1 true, label %bb, label %exit

bb:
  %__global_id_0 = tail call i64 @_Z13get_global_idj(i32 0) #2
  %arrayidx = getelementptr inbounds i64, i64* %A, i64 %__global_id_0
  store i64 0, i64* %arrayidx, align 8, !tbaa !6
  br label %exit

exit:
  ret void
}

; Function Attrs: nounwind readnone
declare i64 @_Z13get_global_idj(i32) local_unnamed_addr #1

attributes #0 = { nounwind uwtable "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-__global_id_0s"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-__global_id_0s"="false" "less-precise-fpmad"="false" "no-frame-pointer-elim"="true" "no-frame-pointer-elim-non-leaf" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+fxsr,+mmx,+sse,+sse2,+x87" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #2 = { nounwind readnone }

!llvm.module.flags = !{!0}
!llvm.ident = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{!"clang version 5.0.0 (trunk 303846) (llvm/trunk 303834)"}
!2 = !{i32 1}
!3 = !{!"none"}
!4 = !{!"long*"}
!5 = !{!""}
!6 = !{!7, !7, i64 0}
!7 = !{!"long", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
