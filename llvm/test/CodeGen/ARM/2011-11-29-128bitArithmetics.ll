; RUN: llc < %s -march=arm -mcpu=cortex-a9 | FileCheck %s

@A = global <4 x float> <float 0., float 1., float 2., float 3.>

define void @test_sqrt(<4 x float>* %X) nounwind {

; CHECK: test_sqrt:

; CHECK:      movw    r1, :lower16:{{.*}}
; CHECK:      movt    r1, :upper16:{{.*}}
; CHECK:      vld1.64 {{.*}}, [r1, :128]
; CHECK:      vsqrt.f32       {{s[0-9]+}}, {{s[0-9]+}}
; CHECK:      vsqrt.f32       {{s[0-9]+}}, {{s[0-9]+}}
; CHECK:      vsqrt.f32       {{s[0-9]+}}, {{s[0-9]+}}
; CHECK:      vsqrt.f32       {{s[0-9]+}}, {{s[0-9]+}}
; CHECK:      vst1.64  {{.*}}

L.entry:
  %0 = load <4 x float>* @A, align 16
  %1 = call <4 x float> @llvm.sqrt.v4f32(<4 x float> %0)
  store <4 x float> %1, <4 x float>* %X, align 16
  ret void
}

declare <4 x float> @llvm.sqrt.v4f32(<4 x float>) nounwind readonly


define void @test_cos(<4 x float>* %X) nounwind {

; CHECK: test_cos:

; CHECK:      movw  [[reg0:r[0-9]+]], :lower16:{{.*}}
; CHECK:      movt  [[reg0]], :upper16:{{.*}}
; CHECK:      vld1.64

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}cosf

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}cosf

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}cosf

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}cosf

; CHECK:      vst1.64

L.entry:
  %0 = load <4 x float>* @A, align 16
  %1 = call <4 x float> @llvm.cos.v4f32(<4 x float> %0)
  store <4 x float> %1, <4 x float>* %X, align 16
  ret void
}

declare <4 x float> @llvm.cos.v4f32(<4 x float>) nounwind readonly

define void @test_exp(<4 x float>* %X) nounwind {

; CHECK: test_exp:

; CHECK:      movw  [[reg0:r[0-9]+]], :lower16:{{.*}}
; CHECK:      movt  [[reg0]], :upper16:{{.*}}
; CHECK:      vld1.64

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}expf

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}expf

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}expf

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}expf

; CHECK:      vst1.64

L.entry:
  %0 = load <4 x float>* @A, align 16
  %1 = call <4 x float> @llvm.exp.v4f32(<4 x float> %0)
  store <4 x float> %1, <4 x float>* %X, align 16
  ret void
}

declare <4 x float> @llvm.exp.v4f32(<4 x float>) nounwind readonly

define void @test_exp2(<4 x float>* %X) nounwind {

; CHECK: test_exp2:

; CHECK:      movw  [[reg0:r[0-9]+]], :lower16:{{.*}}
; CHECK:      movt  [[reg0]], :upper16:{{.*}}
; CHECK:      vld1.64

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}exp2f

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}exp2f

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}exp2f

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}exp2f

; CHECK:      vst1.64

L.entry:
  %0 = load <4 x float>* @A, align 16
  %1 = call <4 x float> @llvm.exp2.v4f32(<4 x float> %0)
  store <4 x float> %1, <4 x float>* %X, align 16
  ret void
}

declare <4 x float> @llvm.exp2.v4f32(<4 x float>) nounwind readonly

define void @test_log10(<4 x float>* %X) nounwind {

; CHECK: test_log10:

; CHECK:      movw  [[reg0:r[0-9]+]], :lower16:{{.*}}
; CHECK:      movt  [[reg0]], :upper16:{{.*}}
; CHECK:      vld1.64

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}log10f

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}log10f

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}log10f

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}log10f

; CHECK:      vst1.64

L.entry:
  %0 = load <4 x float>* @A, align 16
  %1 = call <4 x float> @llvm.log10.v4f32(<4 x float> %0)
  store <4 x float> %1, <4 x float>* %X, align 16
  ret void
}

declare <4 x float> @llvm.log10.v4f32(<4 x float>) nounwind readonly

define void @test_log(<4 x float>* %X) nounwind {

; CHECK: test_log:

; CHECK:      movw  [[reg0:r[0-9]+]], :lower16:{{.*}}
; CHECK:      movt  [[reg0]], :upper16:{{.*}}
; CHECK:      vld1.64

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}logf

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}logf

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}logf

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}logf

; CHECK:      vst1.64

L.entry:
  %0 = load <4 x float>* @A, align 16
  %1 = call <4 x float> @llvm.log.v4f32(<4 x float> %0)
  store <4 x float> %1, <4 x float>* %X, align 16
  ret void
}

declare <4 x float> @llvm.log.v4f32(<4 x float>) nounwind readonly

define void @test_log2(<4 x float>* %X) nounwind {

; CHECK: test_log2:

; CHECK:      movw  [[reg0:r[0-9]+]], :lower16:{{.*}}
; CHECK:      movt  [[reg0]], :upper16:{{.*}}
; CHECK:      vld1.64

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}log2f

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}log2f

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}log2f

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}log2f

; CHECK:      vst1.64

L.entry:
  %0 = load <4 x float>* @A, align 16
  %1 = call <4 x float> @llvm.log2.v4f32(<4 x float> %0)
  store <4 x float> %1, <4 x float>* %X, align 16
  ret void
}

declare <4 x float> @llvm.log2.v4f32(<4 x float>) nounwind readonly


define void @test_pow(<4 x float>* %X) nounwind {

; CHECK: test_pow:

; CHECK:      movw  [[reg0:r[0-9]+]], :lower16:{{.*}}
; CHECK:      movt  [[reg0]], :upper16:{{.*}}
; CHECK:      vld1.64

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}powf

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}powf

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}powf

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}powf

; CHECK:      vst1.64

L.entry:

  %0 = load <4 x float>* @A, align 16
  %1 = call <4 x float> @llvm.pow.v4f32(<4 x float> %0, <4 x float> <float 2., float 2., float 2., float 2.>)

  store <4 x float> %1, <4 x float>* %X, align 16

  ret void
}

declare <4 x float> @llvm.pow.v4f32(<4 x float>, <4 x float>) nounwind readonly

define void @test_powi(<4 x float>* %X) nounwind {

; CHECK: test_powi:

; CHECK:       movw  [[reg0:r[0-9]+]], :lower16:{{.*}}
; CHECK:       movt  [[reg0]], :upper16:{{.*}}
; CHECK:       vld1.64 {{.*}}, :128
; CHECK:       vmul.f32 {{.*}}

; CHECK:      vst1.64

L.entry:

  %0 = load <4 x float>* @A, align 16
  %1 = call <4 x float> @llvm.powi.v4f32(<4 x float> %0, i32 2)

  store <4 x float> %1, <4 x float>* %X, align 16

  ret void
}

declare <4 x float> @llvm.powi.v4f32(<4 x float>, i32) nounwind readonly

define void @test_sin(<4 x float>* %X) nounwind {

; CHECK: test_sin:

; CHECK:      movw  [[reg0:r[0-9]+]], :lower16:{{.*}}
; CHECK:      movt  [[reg0]], :upper16:{{.*}}
; CHECK:      vld1.64

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}sinf

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}sinf

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}sinf

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}sinf

; CHECK:      vst1.64

L.entry:
  %0 = load <4 x float>* @A, align 16
  %1 = call <4 x float> @llvm.sin.v4f32(<4 x float> %0)
  store <4 x float> %1, <4 x float>* %X, align 16
  ret void
}

declare <4 x float> @llvm.sin.v4f32(<4 x float>) nounwind readonly

define void @test_floor(<4 x float>* %X) nounwind {

; CHECK: test_floor:

; CHECK:      movw  [[reg0:r[0-9]+]], :lower16:{{.*}}
; CHECK:      movt  [[reg0]], :upper16:{{.*}}
; CHECK:      vld1.64

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}floorf

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}floorf

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}floorf

; CHECK:      {{v?mov(.32)?}}  r0,
; CHECK:      bl  {{.*}}floorf

; CHECK:      vst1.64

L.entry:
  %0 = load <4 x float>* @A, align 16
  %1 = call <4 x float> @llvm.floor.v4f32(<4 x float> %0)
  store <4 x float> %1, <4 x float>* %X, align 16
  ret void
}

declare <4 x float> @llvm.floor.v4f32(<4 x float>) nounwind readonly

