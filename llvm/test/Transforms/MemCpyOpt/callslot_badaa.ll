; RUN: opt < %s -S -memcpyopt | FileCheck --match-full-lines %s

; Make sure callslot optimization merges alias.scope metadata correctly when it merges instructions.
; Merging here naively generates:
;  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %dst, ptr align 8 %src, i64 1, i1 false), !alias.scope !3
;  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %src), !noalias !0
;   ...
;  !0 = !{!1}
;  !1 = distinct !{!1, !2, !"callee1: %a"}
;  !2 = distinct !{!2, !"callee1"}
;  !3 = !{!1, !4}
;  !4 = distinct !{!4, !5, !"callee0: %a"}
;  !5 = distinct !{!5, !"callee0"}
; Which is incorrect because the lifetime.end of %src will now "noalias" the above memcpy.
define i8 @test(i8 %input) {
  %tmp = alloca i8
  %dst = alloca i8
  %src = alloca i8
; NOTE: we're matching the full line and looking for the lack of !alias.scope here
; CHECK:   call void @llvm.memcpy.p0.p0.i64(ptr align 8 %dst, ptr align 8 %src, i64 1, i1 false)
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %src), !noalias !3
  store i8 %input, ptr %src
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %tmp, ptr align 8 %src, i64 1, i1 false), !alias.scope !0
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %src), !noalias !3
  call void @llvm.memcpy.p0.p0.i64(ptr align 8 %dst, ptr align 8 %tmp, i64 1, i1 false), !alias.scope !3
  %ret_value = load i8, ptr %dst
  ret i8 %ret_value
}

declare void @llvm.lifetime.start.p0(i64, ptr nocapture)
declare void @llvm.lifetime.end.p0(i64, ptr nocapture)
declare void @llvm.memcpy.p0.p0.i64(ptr, ptr, i64, i1)

!0 = !{!1}
!1 = distinct !{!1, !2, !"callee0: %a"}
!2 = distinct !{!2, !"callee0"}
!3 = !{!4}
!4 = distinct !{!4, !5, !"callee1: %a"}
!5 = distinct !{!5, !"callee1"}
