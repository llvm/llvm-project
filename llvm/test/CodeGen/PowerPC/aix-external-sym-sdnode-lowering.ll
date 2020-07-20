; RUN: llc -mcpu=pwr4 -mattr=-altivec -verify-machineinstrs -mtriple powerpc-ibm-aix-xcoff \
; RUN: -stop-after=machine-cp < %s | FileCheck \
; RUN: --check-prefix=32BIT %s

; RUN: llc -mcpu=pwr4 -mattr=-altivec -verify-machineinstrs -mtriple powerpc64-ibm-aix-xcoff \
; RUN: -stop-after=machine-cp < %s | FileCheck \
; RUN: --check-prefix=64BIT %s

define i64 @call_divdi3(i64 %p, i64 %num) {
entry:
  %div = sdiv i64 %p, %num
  ret i64 %div
}

; 32BIT: BL_NOP <mcsymbol .__divdi3>

define i64 @call_fixunsdfdi(double %p) {
entry:
  %conv = fptoui double %p to i64
  ret i64 %conv
}

; 32BIT: BL_NOP <mcsymbol .__fixunsdfdi>

define double @call_floatundidf(i64 %p) {
entry:
  %conv = uitofp i64 %p to double
  ret double %conv
}

; 32BIT: BL_NOP <mcsymbol .__floatundidf>

define float @call_floatundisf(i64 %p) {
entry:
  %conv = uitofp i64 %p to float
  ret float %conv
}

; 32BIT: BL_NOP <mcsymbol .__floatundisf>

define i64 @call_moddi3(i64 %p, i64 %num) {
entry:
  %rem = srem i64 %p, %num
  ret i64 %rem
}

; 32BIT: BL_NOP <mcsymbol .__moddi3>

define i64 @call_udivdi3(i64 %p, i64 %q) {
  %1 = udiv i64 %p, %q
  ret i64 %1
}

; 32BIT: BL_NOP <mcsymbol .__udivdi3>

define i64 @call_umoddi3(i64 %p, i64 %num) {
entry:
  %rem = urem i64 %p, %num
  ret i64 %rem
}

; 32BIT: BL_NOP <mcsymbol .__umoddi3>

define double @call_ceil(double %n) {
entry:
  %0 = call double @llvm.ceil.f64(double %n)
  ret double %0
}

declare double @llvm.ceil.f64(double)

; 32BIT: BL_NOP <mcsymbol .ceil>
; 64BIT: BL8_NOP <mcsymbol .ceil>

define double @call_floor(double %n) {
entry:
  %0 = call double @llvm.floor.f64(double %n)
  ret double %0
}

declare double @llvm.floor.f64(double)

; 32BIT: BL_NOP <mcsymbol .floor>
; 64BIT: BL8_NOP <mcsymbol .floor>

define void @call_memcpy(i8* %p, i8* %q, i32 %n) {
entry:
  call void @llvm.memcpy.p0i8.p0i8.i32(i8* %p, i8* %q, i32 %n, i1 false)
  ret void
}

declare void @llvm.memcpy.p0i8.p0i8.i32(i8* nocapture writeonly, i8* nocapture readonly, i32, i1)

; 32BIT: BL_NOP <mcsymbol .memcpy>
; 64BIT: BL8_NOP <mcsymbol .memcpy>

define void @call_memmove(i8* %p, i8* %q, i32 %n) {
entry:
  call void @llvm.memmove.p0i8.p0i8.i32(i8* %p, i8* %q, i32 %n, i1 false)
  ret void
}

declare void @llvm.memmove.p0i8.p0i8.i32(i8* nocapture, i8* nocapture readonly, i32, i1)

; 32BIT: BL_NOP <mcsymbol .memmove>
; 64BIT: BL8_NOP <mcsymbol .memmove>

define void @call_memset(i8* %p, i8 %q, i32 %n) #0 {
entry:
  call void @llvm.memset.p0i8.i32(i8* %p, i8 %q, i32 %n, i1 false)
  ret void
}

declare void @llvm.memset.p0i8.i32(i8* nocapture, i8, i32, i1)

; 32BIT: BL_NOP <mcsymbol .memset>
; 64BIT: BL8_NOP <mcsymbol .memset>

define double @call_round(double %n) {
entry:
  %0 = call double @llvm.round.f64(double %n)
  ret double %0
}

declare double @llvm.round.f64(double)

; 32BIT: BL_NOP <mcsymbol .round>
; 64BIT: BL8_NOP <mcsymbol .round>
