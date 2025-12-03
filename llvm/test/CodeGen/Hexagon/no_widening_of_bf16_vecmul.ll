;; RUN: llc --mtriple=hexagon --mcpu=hexagonv81 --mattr=+hvxv81,+hvx-length128b %s -o - | FileCheck %s

; In this file, we check that fmul(exttof32(v1.bf16), exttof32(v2.bf16)) is not
; transformed to exttof32(fmul(v1.hf, v2.hf)). This was a bug in
; hexagon-widening-vector pass.

define void @halfbf16(ptr readonly %x, ptr %y) {
entry:
  %xvec.bf16 = load <64 x bfloat>, ptr %x, align 2
  %xvec.f32 = fpext <64 x bfloat> %xvec.bf16 to <64 x float>
  %yvec.f32 = fmul <64 x float> %xvec.f32, splat (float 5.000000e-01)
  %yvec.bf16 = fptrunc <64 x float> %yvec.f32 to <64 x bfloat>
  store <64 x bfloat> %yvec.bf16, ptr %y, align 2
  ret void
}
;; CHECK: vmpy(v{{[0-9]+}}.sf,v{{[0-9]+}}.sf)
;; CHECK: vmpy(v{{[0-9]+}}.sf,v{{[0-9]+}}.sf)


define void @vecmulbf16(ptr readonly %x, ptr readonly %y, ptr %z) {
entry:
  %xvec.bf16 = load <64 x bfloat>, ptr %x, align 2
  %yvec.bf16 = load <64 x bfloat>, ptr %y, align 2
  %xvec.f32 = fpext <64 x bfloat> %xvec.bf16 to <64 x float>
  %yvec.f32 = fpext <64 x bfloat> %yvec.bf16 to <64 x float>
  %zvec.f32 = fmul <64 x float> %xvec.f32, %yvec.f32
  %zvec.bf16 = fptrunc <64 x float> %zvec.f32 to <64 x bfloat>
  store <64 x bfloat> %zvec.bf16, ptr %z, align 2
  ret void
}

;; CHECK: vmpy(v{{[0-9]+}}.sf,v{{[0-9]+}}.sf)
;; CHECK: vmpy(v{{[0-9]+}}.sf,v{{[0-9]+}}.sf)


define void @halff16(ptr readonly %x, ptr %y) {
entry:
  %xvec.f16 = load <64 x half>, ptr %x, align 2
  %xvec.f32 = fpext <64 x half> %xvec.f16 to <64 x float>
  %yvec.f32 = fmul <64 x float> %xvec.f32, splat (float 5.000000e-01)
  %yvec.f16 = fptrunc <64 x float> %yvec.f32 to <64 x half>
  store <64 x half> %yvec.f16, ptr %y, align 2
  ret void
}
;; CHECK: vmpy(v{{[0-9]+}}.hf,v{{[0-9]+}}.hf)


define void @vecmulf16(ptr readonly %x, ptr readonly %y, ptr %z) {
entry:
  %xvec.f16 = load <64 x half>, ptr %x, align 2
  %yvec.f16 = load <64 x half>, ptr %y, align 2
  %xvec.f32 = fpext <64 x half> %xvec.f16 to <64 x float>
  %yvec.f32 = fpext <64 x half> %yvec.f16 to <64 x float>
  %zvec.f32 = fmul <64 x float> %xvec.f32, %yvec.f32
  %zvec.f16 = fptrunc <64 x float> %zvec.f32 to <64 x half>
  store <64 x half> %zvec.f16, ptr %z, align 2
  ret void
}

;; CHECK: vmpy(v{{[0-9]+}}.hf,v{{[0-9]+}}.hf)
