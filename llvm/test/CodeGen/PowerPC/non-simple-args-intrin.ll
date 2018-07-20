; RUN: llc -verify-machineinstrs <%s | FileCheck %s
target datalayout = "e-m:e-i64:64-n32:64"
target triple = "powerpc64le-unknown-linux-gnu"

; Ensure that that the  CTRLoop pass can compile intrinsics with
; non-simple arguments. eg: @llvm.sqrt.v16f64.

; Function Attrs: nounwind
define void @filter_prewitt() {
; CHECK-LABEL: filter_prewitt:
entry:
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %entry
  %wide.load = load <16 x i8>, <16 x i8>* undef, align 1, !tbaa !1, !alias.scope !4
  %0 = zext <16 x i8> %wide.load to <16 x i32>
  %wide.load279 = load <16 x i8>, <16 x i8>* undef, align 1, !tbaa !1, !alias.scope !4
  %1 = zext <16 x i8> %wide.load279 to <16 x i32>
  %2 = add nuw nsw <16 x i32> %1, %0
  %3 = add nuw nsw <16 x i32> %2, zeroinitializer
  %4 = sub nsw <16 x i32> zeroinitializer, %3
  %5 = add nsw <16 x i32> %4, zeroinitializer
  %6 = add nsw <16 x i32> %5, zeroinitializer
  %7 = sub nsw <16 x i32> zeroinitializer, %0
  %8 = sub nsw <16 x i32> %7, zeroinitializer
  %9 = add nsw <16 x i32> %8, zeroinitializer
  %10 = sub nsw <16 x i32> %9, zeroinitializer
  %11 = add nsw <16 x i32> %10, zeroinitializer
  %12 = mul nsw <16 x i32> %6, %6
  %13 = mul nsw <16 x i32> %11, %11
  %14 = add nuw nsw <16 x i32> %13, %12
  %15 = sitofp <16 x i32> %14 to <16 x double>
  %16 = call nsz <16 x double> @llvm.sqrt.v16f64(<16 x double> %15)
  %17 = fmul nsz <16 x double> %16, undef
  %18 = fadd nsz <16 x double> %17, undef
  %19 = fptosi <16 x double> %18 to <16 x i32>
  %20 = sub nsw <16 x i32> zeroinitializer, %19
  %21 = ashr <16 x i32> %20, <i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31, i32 31>
  %22 = select <16 x i1> undef, <16 x i32> %21, <16 x i32> %19
  %23 = trunc <16 x i32> %22 to <16 x i8>
  store <16 x i8> %23, <16 x i8>* undef, align 1, !tbaa !1, !alias.scope !7, !noalias !9
  br label %vector.body
}

; Function Attrs: nounwind readnone speculatable
declare <16 x double> @llvm.sqrt.v16f64(<16 x double>) #1

attributes #1 = { nounwind readnone speculatable }

!1 = !{!2, !2, i64 0}
!2 = !{!"omnipotent char", !3, i64 0}
!3 = !{!"Simple C/C++ TBAA"}
!4 = !{!5}
!5 = distinct !{!5, !6}
!6 = distinct !{!6, !"LVerDomain"}
!7 = !{!8}
!8 = distinct !{!8, !6}
!9 = !{!10, !11, !5}
!10 = distinct !{!10, !6}
!11 = distinct !{!11, !6}
