; RUN: llc -mtriple=arm-eabi -float-abi=soft -mcpu=cortex-a9 %s -o - | FileCheck %s

@A = global <4 x float> <float 0., float 1., float 2., float 3.>

define void @test_sqrt(ptr %X) nounwind {

; CHECK-LABEL: test_sqrt:

; CHECK:      movw    r1, :lower16:{{.*}}
; CHECK:      movt    r1, :upper16:{{.*}}
; CHECK:      vld1.64 {{.*}}, [r1:128]
; CHECK:      vsqrt.f32       {{s[0-9]+}}, {{s[0-9]+}}
; CHECK:      vsqrt.f32       {{s[0-9]+}}, {{s[0-9]+}}
; CHECK:      vsqrt.f32       {{s[0-9]+}}, {{s[0-9]+}}
; CHECK:      vsqrt.f32       {{s[0-9]+}}, {{s[0-9]+}}
; CHECK:      vst1.64  {{.*}}

L.entry:
  %0 = load <4 x float>, ptr @A, align 16
  %1 = call <4 x float> @llvm.sqrt.v4f32(<4 x float> %0)
  store <4 x float> %1, ptr %X, align 16
  ret void
}

declare <4 x float> @llvm.sqrt.v4f32(<4 x float>) nounwind readonly


define void @test_cos(ptr %X) nounwind {

; CHECK-LABEL: test_cos:

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
  %0 = load <4 x float>, ptr @A, align 16
  %1 = call <4 x float> @llvm.cos.v4f32(<4 x float> %0)
  store <4 x float> %1, ptr %X, align 16
  ret void
}

declare <4 x float> @llvm.cos.v4f32(<4 x float>) nounwind readonly

define void @test_exp(ptr %X) nounwind {

; CHECK-LABEL: test_exp:

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
  %0 = load <4 x float>, ptr @A, align 16
  %1 = call <4 x float> @llvm.exp.v4f32(<4 x float> %0)
  store <4 x float> %1, ptr %X, align 16
  ret void
}

declare <4 x float> @llvm.exp.v4f32(<4 x float>) nounwind readonly

define void @test_exp2(ptr %X) nounwind {

; CHECK-LABEL: test_exp2:

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
  %0 = load <4 x float>, ptr @A, align 16
  %1 = call <4 x float> @llvm.exp2.v4f32(<4 x float> %0)
  store <4 x float> %1, ptr %X, align 16
  ret void
}

declare <4 x float> @llvm.exp2.v4f32(<4 x float>) nounwind readonly

define void @test_log10(ptr %X) nounwind {

; CHECK-LABEL: test_log10:

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
  %0 = load <4 x float>, ptr @A, align 16
  %1 = call <4 x float> @llvm.log10.v4f32(<4 x float> %0)
  store <4 x float> %1, ptr %X, align 16
  ret void
}

declare <4 x float> @llvm.log10.v4f32(<4 x float>) nounwind readonly

define void @test_log(ptr %X) nounwind {

; CHECK-LABEL: test_log:

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
  %0 = load <4 x float>, ptr @A, align 16
  %1 = call <4 x float> @llvm.log.v4f32(<4 x float> %0)
  store <4 x float> %1, ptr %X, align 16
  ret void
}

declare <4 x float> @llvm.log.v4f32(<4 x float>) nounwind readonly

define void @test_log2(ptr %X) nounwind {

; CHECK-LABEL: test_log2:

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
  %0 = load <4 x float>, ptr @A, align 16
  %1 = call <4 x float> @llvm.log2.v4f32(<4 x float> %0)
  store <4 x float> %1, ptr %X, align 16
  ret void
}

declare <4 x float> @llvm.log2.v4f32(<4 x float>) nounwind readonly


define void @test_pow(ptr %X) nounwind {

; CHECK-LABEL: test_pow:

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

  %0 = load <4 x float>, ptr @A, align 16
  %1 = call <4 x float> @llvm.pow.v4f32(<4 x float> %0, <4 x float> <float 2., float 2., float 2., float 2.>)

  store <4 x float> %1, ptr %X, align 16

  ret void
}

declare <4 x float> @llvm.pow.v4f32(<4 x float>, <4 x float>) nounwind readonly

define void @test_powi(ptr %X) nounwind {

; CHECK-LABEL: test_powi:

; CHECK:       movw  [[reg0:r[0-9]+]], :lower16:{{.*}}
; CHECK:       movt  [[reg0]], :upper16:{{.*}}
; CHECK:       vld1.64 {{.*}}:128
; CHECK:       vmul.f32 {{.*}}

; CHECK:      vst1.64

L.entry:

  %0 = load <4 x float>, ptr @A, align 16
  %1 = call <4 x float> @llvm.powi.v4f32.i32(<4 x float> %0, i32 2)

  store <4 x float> %1, ptr %X, align 16

  ret void
}

declare <4 x float> @llvm.powi.v4f32.i32(<4 x float>, i32) nounwind readonly

define void @test_sin(ptr %X) nounwind {

; CHECK-LABEL: test_sin:

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
  %0 = load <4 x float>, ptr @A, align 16
  %1 = call <4 x float> @llvm.sin.v4f32(<4 x float> %0)
  store <4 x float> %1, ptr %X, align 16
  ret void
}

declare <4 x float> @llvm.sin.v4f32(<4 x float>) nounwind readonly

define void @test_floor(ptr %X) nounwind {

; CHECK-LABEL: test_floor:

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
  %0 = load <4 x float>, ptr @A, align 16
  %1 = call <4 x float> @llvm.floor.v4f32(<4 x float> %0)
  store <4 x float> %1, ptr %X, align 16
  ret void
}

declare <4 x float> @llvm.floor.v4f32(<4 x float>) nounwind readonly

