; RUN: llc < %s -O0 -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-apple-ios -verify-machineinstrs | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -O0 -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-ios -verify-machineinstrs | FileCheck %s --check-prefix=THUMB
; RUN: llc < %s -O0 -mattr=+strict-align -relocation-model=dynamic-no-pic -mtriple=armv7-apple-ios -verify-machineinstrs | FileCheck %s --check-prefix=ARM-STRICT-ALIGN
; RUN: llc < %s -O0 -mattr=+strict-align -relocation-model=dynamic-no-pic -mtriple=thumbv7-apple-ios -verify-machineinstrs | FileCheck %s --check-prefix=THUMB-STRICT-ALIGN

; RUN: llc < %s -O0 -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-linux-gnueabi -verify-machineinstrs | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -O0 -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=thumbv7-linux-gnueabi -verify-machineinstrs | FileCheck %s --check-prefix=THUMB
; RUN: llc < %s -O0 -mattr=+strict-align -relocation-model=dynamic-no-pic -mtriple=armv7-linux-gnueabi -verify-machineinstrs | FileCheck %s --check-prefix=ARM-STRICT-ALIGN
; RUN: llc < %s -O0 -mattr=+strict-align -relocation-model=dynamic-no-pic -mtriple=thumbv7-linux-gnueabi -verify-machineinstrs | FileCheck %s --check-prefix=THUMB-STRICT-ALIGN

; RUN: llc < %s -O0 -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-unknown-nacl -verify-machineinstrs | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -O0 -mattr=+strict-align -relocation-model=dynamic-no-pic -mtriple=armv7-unknown-nacl -verify-machineinstrs | FileCheck %s --check-prefix=ARM-STRICT-ALIGN

; RUN: llc < %s -O0 -mattr=+strict-align -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-unknown-unknown -verify-machineinstrs | FileCheck %s --check-prefix=ARM-STRICT-ALIGN
; RUN: llc < %s -O0  -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=thumbv7-unknown-unknown -mattr=+strict-align -verify-machineinstrs | FileCheck %s --check-prefix=THUMB-STRICT-ALIGN
; RUN: llc < %s -O0 -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=armv7-unknown-unknown -verify-machineinstrs | FileCheck %s --check-prefix=ARM
; RUN: llc < %s -O0 -fast-isel-abort=1 -relocation-model=dynamic-no-pic -mtriple=thumbv7-unknown-unknown -verify-machineinstrs | FileCheck %s --check-prefix=THUMB
; RUN: llc < %s -O0 -relocation-model=dynamic-no-pic -mtriple=armv7-unknown-unknown -mattr=+strict-align -verify-machineinstrs | FileCheck %s --check-prefix=ARM-STRICT-ALIGN
; RUN: llc < %s -O0 -relocation-model=dynamic-no-pic -mtriple=thumbv7-unknown-unknown -mattr=+strict-align -verify-machineinstrs | FileCheck %s --check-prefix=THUMB-STRICT-ALIGN

; Check unaligned stores
%struct.anon = type <{ float }>

@a = common global %struct.anon* null, align 4

define void @unaligned_store(float %x, float %y) nounwind {
entry:
; ARM: @unaligned_store
; ARM: vmov r1, s0
; ARM: str r1, [r0]

; THUMB: @unaligned_store
; THUMB: vmov r1, s0
; THUMB: str r1, [r0]

  %add = fadd float %x, %y
  %0 = load %struct.anon*, %struct.anon** @a, align 4
  %x1 = getelementptr inbounds %struct.anon, %struct.anon* %0, i32 0, i32 0
  store float %add, float* %x1, align 1
  ret void
}

; Doublewords require only word-alignment.
; rdar://10528060
%struct.anon.0 = type { double }

@foo_unpacked = common global %struct.anon.0 zeroinitializer, align 4

define void @word_aligned_f64_store(double %a, double %b) nounwind {
entry:
; ARM: @word_aligned_f64_store
; THUMB: @word_aligned_f64_store
  %add = fadd double %a, %b
  store double %add, double* getelementptr inbounds (%struct.anon.0, %struct.anon.0* @foo_unpacked, i32 0, i32 0), align 4
; ARM: vstr d16, [r0]
; THUMB: vstr d16, [r0]
  ret void
}

; Check unaligned loads of floats
%class.TAlignTest = type <{ i16, float }>

define zeroext i1 @unaligned_f32_load(%class.TAlignTest* %this) nounwind align 2 {
entry:
; ARM: @unaligned_f32_load
; THUMB: @unaligned_f32_load
  %0 = alloca %class.TAlignTest*, align 4
  store %class.TAlignTest* %this, %class.TAlignTest** %0, align 4
  %1 = load %class.TAlignTest*, %class.TAlignTest** %0
  %2 = getelementptr inbounds %class.TAlignTest, %class.TAlignTest* %1, i32 0, i32 1
  %3 = load float, float* %2, align 1
  %4 = fcmp une float %3, 0.000000e+00
; ARM: ldr r[[R:[0-9]+]], [r0, #2]
; ARM: vmov s0, r[[R]]
; ARM: vcmp.f32 s0, #0
; THUMB: ldr.w r[[R:[0-9]+]], [r0, #2]
; THUMB: vmov s0, r[[R]]
; THUMB: vcmp.f32 s0, #0
  ret i1 %4
}

define void @unaligned_i16_store(i16 %x, i16* %y) nounwind {
entry:
; ARM-STRICT-ALIGN: @unaligned_i16_store
; ARM-STRICT-ALIGN: strb
; ARM-STRICT-ALIGN: strb

; THUMB-STRICT-ALIGN: @unaligned_i16_store
; THUMB-STRICT-ALIGN: strb
; THUMB-STRICT-ALIGN: strb

  store i16 %x, i16* %y, align 1
  ret void
}

define i16 @unaligned_i16_load(i16* %x) nounwind {
entry:
; ARM-STRICT-ALIGN: @unaligned_i16_load
; ARM-STRICT-ALIGN: ldrb
; ARM-STRICT-ALIGN: ldrb

; THUMB-STRICT-ALIGN: @unaligned_i16_load
; THUMB-STRICT-ALIGN: ldrb
; THUMB-STRICT-ALIGN: ldrb

  %0 = load i16, i16* %x, align 1
  ret i16 %0
}

define void @unaligned_i32_store(i32 %x, i32* %y) nounwind {
entry:
; ARM-STRICT-ALIGN: @unaligned_i32_store
; ARM-STRICT-ALIGN: strb
; ARM-STRICT-ALIGN: strb
; ARM-STRICT-ALIGN: strb
; ARM-STRICT-ALIGN: strb

; THUMB-STRICT-ALIGN: @unaligned_i32_store
; THUMB-STRICT-ALIGN: strb
; THUMB-STRICT-ALIGN: strb
; THUMB-STRICT-ALIGN: strb
; THUMB-STRICT-ALIGN: strb

  store i32 %x, i32* %y, align 1
  ret void
}

define i32 @unaligned_i32_load(i32* %x) nounwind {
entry:
; ARM-STRICT-ALIGN: @unaligned_i32_load
; ARM-STRICT-ALIGN: ldrb
; ARM-STRICT-ALIGN: ldrb
; ARM-STRICT-ALIGN: ldrb
; ARM-STRICT-ALIGN: ldrb

; THUMB-STRICT-ALIGN: @unaligned_i32_load
; THUMB-STRICT-ALIGN: ldrb
; THUMB-STRICT-ALIGN: ldrb
; THUMB-STRICT-ALIGN: ldrb
; THUMB-STRICT-ALIGN: ldrb

  %0 = load i32, i32* %x, align 1
  ret i32 %0
}
