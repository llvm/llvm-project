; RUN: opt < %s -aa-pipeline=basic-aa -passes=aa-eval -print-all-alias-modref-info -disable-output 2>&1 | FileCheck %s
; REQUIRES: arm-registered-target

target datalayout = "e-p:32:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:32:32-f32:32:32-f64:32:32-v64:32:64-v128:32:128-a0:0:32-n32"
target triple = "arm-apple-ios"

declare <8 x i16> @llvm.arm.neon.vld1.v8i16.p0i8(i8*, i32) nounwind readonly
declare void @llvm.arm.neon.vst1.p0i8.v8i16(i8*, <8 x i16>, i32) nounwind

define <8 x i16> @test1(i8* %p, <8 x i16> %y) {
entry:
  %q = getelementptr i8, i8* %p, i64 16
  %a = call <8 x i16> @llvm.arm.neon.vld1.v8i16.p0i8(i8* %p, i32 16) nounwind
  call void @llvm.arm.neon.vst1.p0i8.v8i16(i8* %q, <8 x i16> %y, i32 16)
  %b = call <8 x i16> @llvm.arm.neon.vld1.v8i16.p0i8(i8* %p, i32 16) nounwind
  %c = add <8 x i16> %a, %b
  load i8, i8* %p
  load i8, i8* %q
  ret <8 x i16> %c

; CHECK-LABEL: Function: test1:

; CHECK: NoAlias:      i8* %p, i8* %q
; CHECK: Just Ref:  Ptr: i8* %p        <->  %a = call <8 x i16> @llvm.arm.neon.vld1.v8i16.p0i8(i8* %p, i32 16)
; CHECK: NoModRef:  Ptr: i8* %q        <->  %a = call <8 x i16> @llvm.arm.neon.vld1.v8i16.p0i8(i8* %p, i32 16)
; CHECK: NoModRef:  Ptr: i8* %p        <->  call void @llvm.arm.neon.vst1.p0i8.v8i16(i8* %q, <8 x i16> %y, i32 16)
; CHECK: Both ModRef:  Ptr: i8* %q     <->  call void @llvm.arm.neon.vst1.p0i8.v8i16(i8* %q, <8 x i16> %y, i32 16)
; CHECK: Just Ref:  Ptr: i8* %p        <->  %b = call <8 x i16> @llvm.arm.neon.vld1.v8i16.p0i8(i8* %p, i32 16)
; CHECK: NoModRef:  Ptr: i8* %q        <->  %b = call <8 x i16> @llvm.arm.neon.vld1.v8i16.p0i8(i8* %p, i32 16)
; CHECK: NoModRef:   %a = call <8 x i16> @llvm.arm.neon.vld1.v8i16.p0i8(i8* %p, i32 16) #{{[0-9]+}} <->   call void @llvm.arm.neon.vst1.p0i8.v8i16(i8* %q, <8 x i16> %y, i32 16)
; CHECK: NoModRef:   %a = call <8 x i16> @llvm.arm.neon.vld1.v8i16.p0i8(i8* %p, i32 16) #{{[0-9]+}} <->   %b = call <8 x i16> @llvm.arm.neon.vld1.v8i16.p0i8(i8* %p, i32 16)
; CHECK: NoModRef:   call void @llvm.arm.neon.vst1.p0i8.v8i16(i8* %q, <8 x i16> %y, i32 16) <->   %a = call <8 x i16> @llvm.arm.neon.vld1.v8i16.p0i8(i8* %p, i32 16)
; CHECK: NoModRef:   call void @llvm.arm.neon.vst1.p0i8.v8i16(i8* %q, <8 x i16> %y, i32 16) <->   %b = call <8 x i16> @llvm.arm.neon.vld1.v8i16.p0i8(i8* %p, i32 16)
; CHECK: NoModRef:   %b = call <8 x i16> @llvm.arm.neon.vld1.v8i16.p0i8(i8* %p, i32 16) #{{[0-9]+}} <->   %a = call <8 x i16> @llvm.arm.neon.vld1.v8i16.p0i8(i8* %p, i32 16)
; CHECK: NoModRef:   %b = call <8 x i16> @llvm.arm.neon.vld1.v8i16.p0i8(i8* %p, i32 16) #{{[0-9]+}} <->   call void @llvm.arm.neon.vst1.p0i8.v8i16(i8* %q, <8 x i16> %y, i32 16)
}
