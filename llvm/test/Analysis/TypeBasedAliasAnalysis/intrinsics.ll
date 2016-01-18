; RUN: opt -tbaa -basicaa -gvn -S < %s | FileCheck %s

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"

; TBAA should prove that these calls don't interfere, since they are
; IntrArgReadMem and have TBAA metadata.

; CHECK:      define <8 x i16> @test0(i8* %p, i8* %q, <8 x i16> %y) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %a = call <8 x i16> @llvm.arm.neon.vld1.v8i16.p0i8(i8* %p, i32 16) [[NUW:#[0-9]+]]
; CHECK-NEXT:   call void @llvm.arm.neon.vst1.p0i8.v8i16(i8* %q, <8 x i16> %y, i32 16)
; CHECK-NEXT:   %c = add <8 x i16> %a, %a
define <8 x i16> @test0(i8* %p, i8* %q, <8 x i16> %y) {
entry:
  %a = call <8 x i16> @llvm.arm.neon.vld1.v8i16.p0i8(i8* %p, i32 16) nounwind, !tbaa !2
  call void @llvm.arm.neon.vst1.p0i8.v8i16(i8* %q, <8 x i16> %y, i32 16), !tbaa !1
  %b = call <8 x i16> @llvm.arm.neon.vld1.v8i16.p0i8(i8* %p, i32 16) nounwind, !tbaa !2
  %c = add <8 x i16> %a, %b
  ret <8 x i16> %c
}

declare <8 x i16> @llvm.arm.neon.vld1.v8i16.p0i8(i8*, i32) nounwind readonly
declare void @llvm.arm.neon.vst1.p0i8.v8i16(i8*, <8 x i16>, i32) nounwind

; CHECK: attributes #0 = { nounwind readonly argmemonly }
; CHECK: attributes #1 = { nounwind argmemonly }
; CHECK: attributes [[NUW]] = { nounwind }

!0 = !{!"tbaa root", null}
!1 = !{!3, !3, i64 0}
!2 = !{!4, !4, i64 0}
!3 = !{!"A", !0}
!4 = !{!"B", !0}
