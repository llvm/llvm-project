; RUN: opt -aa-pipeline=basic-aa -passes=gvn -S < %s | FileCheck %s
; REQUIRES: arm-registered-target

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"

; BasicAA should prove that these calls don't interfere, since we've
; specifically special cased exactly these two intrinsics in
; MemoryLocation::getForArgument.

; CHECK:      define <8 x i16> @test1(i8* %p, <8 x i16> %y) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %q = getelementptr i8, i8* %p, i64 16
; CHECK-NEXT:   %a = call <8 x i16> @llvm.arm.neon.vld1.v8i16.p0i8(i8* %p, i32 16) [[ATTR:#[0-9]+]]
; CHECK-NEXT:   call void @llvm.arm.neon.vst1.p0i8.v8i16(i8* %q, <8 x i16> %y, i32 16)
; CHECK-NEXT:   %c = add <8 x i16> %a, %a
define <8 x i16> @test1(i8* %p, <8 x i16> %y) {
entry:
  %q = getelementptr i8, i8* %p, i64 16
  %a = call <8 x i16> @llvm.arm.neon.vld1.v8i16.p0i8(i8* %p, i32 16) nounwind
  call void @llvm.arm.neon.vst1.p0i8.v8i16(i8* %q, <8 x i16> %y, i32 16)
  %b = call <8 x i16> @llvm.arm.neon.vld1.v8i16.p0i8(i8* %p, i32 16) nounwind
  %c = add <8 x i16> %a, %b
  ret <8 x i16> %c
}

declare <8 x i16> @llvm.arm.neon.vld1.v8i16.p0i8(i8*, i32) nounwind readonly
declare void @llvm.arm.neon.vst1.p0i8.v8i16(i8*, <8 x i16>, i32) nounwind

; CHECK: attributes #0 = { argmemonly nocallback nofree nosync nounwind readonly willreturn }
; CHECK: attributes #1 = { argmemonly nocallback nofree nosync nounwind willreturn }
; CHECK: attributes [[ATTR]] = { nounwind }
