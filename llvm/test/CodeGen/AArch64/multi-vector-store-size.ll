; RUN: llc -mtriple=aarch64-linux-gnu -stop-after=instruction-select < %s | FileCheck %s

declare void @llvm.aarch64.neon.st2.v4f32.p0f32(<4 x float>, <4 x float>, float*)
declare void @llvm.aarch64.neon.st3.v4f32.p0f32(<4 x float>, <4 x float>, <4 x float>, float*)
declare void @llvm.aarch64.neon.st4.v4f32.p0f32(<4 x float>, <4 x float>, <4 x float>, <4 x float>, float*)

declare void @llvm.aarch64.neon.st1x2.v4f32.p0f32(<4 x float>, <4 x float>, float*)
declare void @llvm.aarch64.neon.st1x3.v4f32.p0f32(<4 x float>, <4 x float>, <4 x float>, float*)
declare void @llvm.aarch64.neon.st1x4.v4f32.p0f32(<4 x float>, <4 x float>, <4 x float>, <4 x float>, float*)

declare void @llvm.aarch64.neon.st2lane.v4f32.p0f32(<4 x float>, <4 x float>, i64, float*)
declare void @llvm.aarch64.neon.st3lane.v4f32.p0f32(<4 x float>, <4 x float>, <4 x float>, i64, float*)
declare void @llvm.aarch64.neon.st4lane.v4f32.p0f32(<4 x float>, <4 x float>, <4 x float>, <4 x float>, i64, float*)

define void @addstx(float* %res, <4 x float>* %a,  <4 x float>* %b, <4 x float>* %c, <4 x float>* %d) {
  %al = load <4 x float>, <4 x float>* %a
  %bl = load <4 x float>, <4 x float>* %b
  %cl = load <4 x float>, <4 x float>* %c
  %dl = load <4 x float>, <4 x float>* %d

  %ar = fadd <4 x float> %al, %bl
  %br = fadd <4 x float> %bl, %cl
  %cr = fadd <4 x float> %cl, %dl
  %dr = fadd <4 x float> %dl, %al

; The sizes below are conservative.  AArch64TargetLowering
; conservatively assumes the entire vector is stored.
  tail call void @llvm.aarch64.neon.st2.v4f32.p0f32(<4 x float> %ar, <4 x float> %br, float* %res)
; CHECK: ST2Twov4s {{.*}} :: (store (s256) {{.*}})
  tail call void @llvm.aarch64.neon.st3.v4f32.p0f32(<4 x float> %ar, <4 x float> %br, <4 x float> %cr, float* %res)
; CHECK: ST3Threev4s {{.*}} :: (store (s384) {{.*}})
  tail call void @llvm.aarch64.neon.st4.v4f32.p0f32(<4 x float> %ar, <4 x float> %br, <4 x float> %cr, <4 x float> %dr, float* %res)
; CHECK: ST4Fourv4s {{.*}} :: (store (s512) {{.*}})

  ret void
}

define void @addst1x(float* %res, <4 x float>* %a,  <4 x float>* %b, <4 x float>* %c, <4 x float>* %d) {
  %al = load <4 x float>, <4 x float>* %a
  %bl = load <4 x float>, <4 x float>* %b
  %cl = load <4 x float>, <4 x float>* %c
  %dl = load <4 x float>, <4 x float>* %d

  %ar = fadd <4 x float> %al, %bl
  %br = fadd <4 x float> %bl, %cl
  %cr = fadd <4 x float> %cl, %dl
  %dr = fadd <4 x float> %dl, %al

; The sizes below are conservative.  AArch64TargetLowering
; conservatively assumes the entire vector is stored.
  tail call void @llvm.aarch64.neon.st1x2.v4f32.p0f32(<4 x float> %ar, <4 x float> %br, float* %res)
; CHECK: ST1Twov4s {{.*}} :: (store (s256) {{.*}})
  tail call void @llvm.aarch64.neon.st1x3.v4f32.p0f32(<4 x float> %ar, <4 x float> %br, <4 x float> %cr, float* %res)
; CHECK: ST1Threev4s {{.*}} :: (store (s384) {{.*}})
  tail call void @llvm.aarch64.neon.st1x4.v4f32.p0f32(<4 x float> %ar, <4 x float> %br, <4 x float> %cr, <4 x float> %dr, float* %res)
; CHECK: ST1Fourv4s {{.*}} :: (store (s512) {{.*}})

  ret void
}

define void @addstxlane(float* %res, <4 x float>* %a,  <4 x float>* %b, <4 x float>* %c, <4 x float>* %d) {
  %al = load <4 x float>, <4 x float>* %a
  %bl = load <4 x float>, <4 x float>* %b
  %cl = load <4 x float>, <4 x float>* %c
  %dl = load <4 x float>, <4 x float>* %d

  %ar = fadd <4 x float> %al, %bl
  %br = fadd <4 x float> %bl, %cl
  %cr = fadd <4 x float> %cl, %dl
  %dr = fadd <4 x float> %dl, %al

; The sizes below are conservative.  AArch64TargetLowering
; conservatively assumes the entire vector is stored.
  tail call void @llvm.aarch64.neon.st2lane.v4f32.p0f32(<4 x float> %ar, <4 x float> %br, i64 1, float* %res)
; CHECK: ST2i32 {{.*}} :: (store (s256) {{.*}})
  tail call void @llvm.aarch64.neon.st3lane.v4f32.p0f32(<4 x float> %ar, <4 x float> %br, <4 x float> %cr, i64 1, float* %res)
; CHECK: ST3i32 {{.*}} :: (store (s384) {{.*}})
  tail call void @llvm.aarch64.neon.st4lane.v4f32.p0f32(<4 x float> %ar, <4 x float> %br, <4 x float> %cr, <4 x float> %dr, i64 1, float* %res)
; CHECK: ST4i32 {{.*}} :: (store (s512) {{.*}})

  ret void
}
