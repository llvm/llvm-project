; RUN: llc -mtriple=thumbv7-linux-gnueabi -o - %s | FileCheck %s

; This alloca is just large enough that FrameLowering decides it needs a frame
; to guarantee access, based on the range of ldrex.

; The actual alloca size is a bit of black magic, unfortunately: the real
; maximum accessible is 1020, but FrameLowering adds 16 bytes to its estimated
; stack size just because so the alloca is not actually the what the limit gets
; compared to. The important point is that we don't go up to ~4096, which is the
; default with no strange instructions.
define void @test_large_frame() {
; CHECK-LABEL: test_large_frame:
; CHECK: push
; CHECK: sub.w sp, sp, #1008

  %ptr = alloca i32, i32 252

  %addr = getelementptr i32, i32* %ptr, i32 1
  call i32 @llvm.arm.ldrex.p0i32(i32* %addr)
  ret void
}

; This alloca is just is just the other side of the limit, so no frame
define void @test_small_frame() {
; CHECK-LABEL: test_small_frame:
; CHECK-NOT: push
; CHECK: sub.w sp, sp, #1004

  %ptr = alloca i32, i32 251

  %addr = getelementptr i32, i32* %ptr, i32 1
  call i32 @llvm.arm.ldrex.p0i32(i32* %addr)
  ret void
}

declare i32 @llvm.arm.ldrex.p0i32(i32*)
