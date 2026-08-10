; RUN: opt -aa-pipeline=basic-aa -passes=gvn -S < %s | FileCheck %s
; REQUIRES: arm-registered-target

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"

; BasicAA should prove that these calls don't interfere, since we've
; specifically special cased exactly these two intrinsics in
; MemoryLocation::getForArgument.

; CHECK:      define <8 x i16> @test1(ptr %p, <8 x i16> %y) {
; CHECK-NEXT: entry:
; CHECK-NEXT:   %q = getelementptr i8, ptr %p, i64 16
; CHECK-NEXT:   %a = call <8 x i16> @llvm.arm.neon.vld1.v8i16.p0(ptr %p, i32 16) [[ATTR:#[0-9]+]]
; CHECK-NEXT:   call void @llvm.arm.neon.vst1.p0.v8i16(ptr %q, <8 x i16> %y, i32 16)
; CHECK-NEXT:   %c = add <8 x i16> %a, %a
define <8 x i16> @test1(ptr %p, <8 x i16> %y) {
entry:
  %q = getelementptr i8, ptr %p, i64 16
  %a = call <8 x i16> @llvm.arm.neon.vld1.v8i16.p0(ptr %p, i32 16) nounwind
  call void @llvm.arm.neon.vst1.p0.v8i16(ptr %q, <8 x i16> %y, i32 16)
  %b = call <8 x i16> @llvm.arm.neon.vld1.v8i16.p0(ptr %p, i32 16) nounwind
  %c = add <8 x i16> %a, %b
  ret <8 x i16> %c
}

declare <8 x i16> @llvm.arm.neon.vld1.v8i16.p0(ptr, i32) nounwind readonly
declare void @llvm.arm.neon.vst1.p0.v8i16(ptr, <8 x i16>, i32) nounwind

; CHECK: attributes #0 = { nocallback nofree nosync nounwind willreturn memory(argmem: read) }
; CHECK: attributes #1 = { nocallback nofree nosync nounwind willreturn memory(argmem: write) }
; CHECK: attributes [[ATTR]] = { nounwind }
