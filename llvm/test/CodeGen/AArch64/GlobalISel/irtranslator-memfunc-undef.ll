; RUN: llc -mtriple=aarch64-unknown-unknown -global-isel -verify-machineinstrs -stop-after=irtranslator %s -o - | FileCheck %s

define void @memset() {
  ; CHECK-LABEL: name: memset
  ; CHECK: bb.1.entry:
  ; CHECK-NEXT:   [[FRAME_INDEX:%[0-9]+]]:_(p0) = G_FRAME_INDEX %stack.0.buf
  ; CHECK-NEXT:   RET_ReallyLR
entry:
  %buf = alloca [512 x i8], align 1
  %ptr = getelementptr inbounds [512 x i8], [512 x i8]* %buf, i32 0, i32 0
  call void @llvm.memset.p0i8.i32(i8* %ptr, i8 undef, i32 512, i1 false)
  ret void
}

define void @memcpy() {
  ; CHECK-LABEL: name: memcpy
  ; CHECK: bb.1.entry:
  ; CHECK-NEXT:   [[FRAME_INDEX:%[0-9]+]]:_(p0) = G_FRAME_INDEX %stack.0.buf
  ; CHECK-NEXT:   RET_ReallyLR
entry:
  %buf = alloca [512 x i8], align 1
  %ptr = getelementptr inbounds [512 x i8], [512 x i8]* %buf, i32 0, i32 0
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %ptr, i8* undef, i32 512, i1 false)
  ret void
}

define void @memmove() {
  ; CHECK-LABEL: name: memmove
  ; CHECK: bb.1.entry:
  ; CHECK-NEXT: [[FRAME_INDEX:%[0-9]+]]:_(p0) = G_FRAME_INDEX %stack.0.buf
  ; CHECK-NEXT: RET_ReallyLR
entry:
  %buf = alloca [512 x i8], align 1
  %ptr = getelementptr inbounds [512 x i8], [512 x i8]* %buf, i32 0, i32 0
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %ptr, i8* undef, i32 512, i1 false)
  ret void
}

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i1) nounwind
declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture writeonly, i8* nocapture readonly, i32, i1) nounwind
declare void @llvm.memmove.p0i8.p0i8.i32(i8* nocapture, i8* nocapture, i32, i1) nounwind
